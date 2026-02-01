import logging
import time
from statistics import mean, stdev

import hydra
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy

from src.config.main_config import MainConfig
from src.construct_model import construct_model
from src.datasets.construct_dataset import get_dataloaders, get_dataset
from src.datasets.transforms import get_transforms_from_config
from src.models.codebook_wrappers import create_codebook_wrapper
from src.training import train_epoch, train_epoch_cosine_codebook
from src.utils import construct_init_function, create_optimizers, set_reproducibility

logger = logging.getLogger(__name__)


def _build_dataloaders(cfg: MainConfig):
    train_transform, val_transform = get_transforms_from_config(
        cfg.dataset,
        model_name=cfg.model.name,
    )
    train_ds, val_ds = get_dataset(
        name=cfg.dataset.name,
        train_transform=train_transform,
        val_transform=val_transform,
        path=cfg.dataset._path,
    )
    train_dl, val_dl = get_dataloaders(
        train_dl_config=cfg.train_dataloader,
        val_dl_config=cfg.val_dataloader,
        train_dataset=train_ds,
        val_dataset=val_ds,
        train_sampler=None,
        val_sampler=None,
    )
    return train_ds, train_dl, val_dl


def _build_mixup_and_criterion(cfg: MainConfig) -> tuple[Mixup | None, nn.Module]:
    if cfg.training.use_mixup:
        mixup_fn = Mixup(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            cutmix_minmax=None,
            prob=1.0,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=cfg.training.label_smoothing,
            num_classes=cfg.dataset.num_classes,
        )
        criterion: nn.Module = SoftTargetCrossEntropy()
    else:
        mixup_fn = None
        criterion = nn.CrossEntropyLoss()

    return mixup_fn, criterion


def _time_epoch(fn, device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - start


def _run_full_training(
    cfg: MainConfig,
    device: torch.device,
    train_dataloader: torch.utils.data.DataLoader,
    epochs: int,
) -> list[float]:
    logger.info("Running full training (unfrozen backbone/head, no codebook)")
    model = construct_model(cfg, device)
    for param in model.parameters():
        param.requires_grad = True
    model = model.to(device)

    if cfg.base_optimizer._target_.startswith("[Warning]"):
        raise ValueError("base_optimizer is not set. Override base_optimizer=adamw/sgd.")

    optimizer = hydra.utils.instantiate(cfg.base_optimizer, model.parameters())
    mixup_fn, criterion = _build_mixup_and_criterion(cfg)

    if cfg.training.compile_model:
        logger.info("Compiling model for full training")
        torch.set_float32_matmul_precision("high")
        model.compile(mode=cfg.training.compile_mode)

    durations: list[float] = []
    for epoch in range(epochs):
        duration = _time_epoch(
            lambda: train_epoch(
                model=model,
                train_dataloader=train_dataloader,
                transforms=mixup_fn,
                optimizers=[optimizer],
                criterion=criterion,
                device=device,
                schedulers=None,
                wandb_run=None,
                epoch=epoch,
            ),
            device,
        )
        logger.info("Full training epoch %s took %.3fs", epoch, duration)
        durations.append(duration)

    return durations


def _run_codebook_training(
    cfg: MainConfig,
    device: torch.device,
    train_dataloader: torch.utils.data.DataLoader,
    epochs: int,
) -> list[float]:
    logger.info("Running codebook training (frozen backbone/head)")
    model = construct_model(cfg, device)
    codebook = hydra.utils.instantiate(cfg.codebook).to(device)

    if hasattr(codebook, "initialize_embeddings"):
        init_function = construct_init_function(cfg.codebook_init)
        codebook.initialize_embeddings(init_func=init_function)

    if cfg.codebook_path:
        codebook.load_state_dict(
            torch.load(cfg.codebook_path, map_location=device, weights_only=True)
        )

    model_with_codebook = create_codebook_wrapper(
        model=model,
        codebook=codebook,
        model_name=cfg.model.name,
        unfreeze_before=0,
    ).to(device)

    optimizers = create_optimizers(
        model=model_with_codebook,
        codebook=codebook,
        cfg=cfg,
    )
    mixup_fn, criterion = _build_mixup_and_criterion(cfg)
    scaler = torch.amp.GradScaler(
        device=device.type,
        enabled=cfg.training.use_amp,
    )

    if cfg.training.compile_model:
        logger.info("Compiling model for codebook training")
        torch.set_float32_matmul_precision("high")
        model_with_codebook.compile(mode=cfg.training.compile_mode)

    durations: list[float] = []
    for epoch in range(epochs):
        duration = _time_epoch(
            lambda: train_epoch_cosine_codebook(
                model=model_with_codebook,
                train_dataloader=train_dataloader,
                num_classes=cfg.dataset.num_classes,
                transforms=mixup_fn,
                optimizers=optimizers,
                schedulers=[],
                criterion=criterion,
                task_loss_weight=cfg.training.task_loss_weight,
                codebook_loss_weight=cfg.training.codebook_loss_weight,
                device=device,
                scaler=scaler,
                wandb_run=None,
            ),
            device,
        )
        logger.info("Codebook training epoch %s took %.3fs", epoch, duration)
        durations.append(duration)

    return durations


def _average_after_warmup(times: list[float], warmup_epochs: int) -> float:
    if not times:
        return 0.0
    if warmup_epochs >= len(times):
        warmup_epochs = 0
    trimmed = times[warmup_epochs:]
    return mean(trimmed)


def _stdev_after_warmup(times: list[float], warmup_epochs: int) -> float:
    if len(times) < 2:
        return 0.0
    if warmup_epochs >= len(times):
        warmup_epochs = 0
    trimmed = times[warmup_epochs:]
    if len(trimmed) < 2:
        return 0.0
    return stdev(trimmed)


@hydra.main(config_path="config", config_name="main_config", version_base="1.2")
def start_compare_epoch_time(cfg: MainConfig) -> None:
    logger.setLevel(cfg._logging_level)
    logger.info(OmegaConf.to_yaml(cfg))

    set_reproducibility(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, train_dataloader, _ = _build_dataloaders(cfg)
    logger.info("Training dataset size: %s", len(train_ds))

    epochs = cfg.timing.epochs if cfg.timing.epochs > 0 else cfg.epochs
    if epochs <= 0:
        raise ValueError("epochs must be > 0 (set cfg.epochs or timing.epochs)")

    warmup = max(0, cfg.timing.warmup_epochs)
    mode = cfg.timing.mode.lower()

    full_times: list[float] = []
    codebook_times: list[float] = []

    if mode in ("full", "both"):
        full_times = _run_full_training(cfg, device, train_dataloader, epochs)
        avg_full = _average_after_warmup(full_times, warmup)
        std_full = _stdev_after_warmup(full_times, warmup)
        logger.info("Avg full training epoch time: %.3fs (±%.3fs)", avg_full, std_full)

    if mode in ("codebook", "both"):
        codebook_times = _run_codebook_training(cfg, device, train_dataloader, epochs)
        avg_codebook = _average_after_warmup(codebook_times, warmup)
        std_codebook = _stdev_after_warmup(codebook_times, warmup)
        logger.info(
            "Avg codebook training epoch time: %.3fs (±%.3fs)",
            avg_codebook,
            std_codebook,
        )

    if mode == "both" and full_times and codebook_times:
        avg_full = _average_after_warmup(full_times, warmup)
        avg_codebook = _average_after_warmup(codebook_times, warmup)
        saved = avg_full - avg_codebook
        saved_pct = (saved / avg_full * 100.0) if avg_full > 0 else 0.0
        logger.info("Average time saved per epoch: %.3fs (%.1f%%)", saved, saved_pct)


if __name__ == "__main__":
    start_compare_epoch_time()
