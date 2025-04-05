from pathlib import Path
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from src.codebook import create_codebook_wrapper
from src.construct_model import construct_model
from src.datasets.construct_dataset import get_dataloaders
from src.utils import (
    train_epoch_cosine_codebook,
    validate_epoch,
    validate_epoch_cosine_codebook,
    set_reproducibility,
    save_checkpoint,
    create_optimizers,
)
from datetime import datetime
import logging
import hydra
from src.config.main_config import MainConfig
from torchvision.transforms import v2 as transforms_v2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None
    logger.info("wandb is not available, skipping wandb.init")


def codebook_training(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    train_transforms: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[torch.optim.lr_scheduler._LRScheduler],
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    wandb_run=None,
):
    for epoch in range(epochs):
        logger.info(f"Epoch: {epoch}")
        codebook_statistics = train_epoch_cosine_codebook(
            model=model,
            train_dataloader=train_dataloader,
            transforms=train_transforms,
            optimizers=optimizers,
            schedulers=schedulers,
            criterion=criterion,
            device=device,
            wandb_run=wandb_run,
        )

        top1_acc, top5_acc = validate_epoch_cosine_codebook(
            model=model, val_dataloader=val_dataloader, device=device
        )

        logger.info(f"Validation top1-accuracy {top1_acc}, top5-accuracy {top5_acc}")
        logger.info(f"Codebook statistics: {codebook_statistics}")
        if wandb_run:
            wandb.log(
                {
                    "Epoch": epoch,
                    "Validation Top1 Accuracy": top1_acc,
                    "Validation Top5 Accuracy": top5_acc,
                    **codebook_statistics,
                }
            )


def prepare_codebook_training(
    cfg: MainConfig, device: torch.device, wandb_run=None
) -> None:
    """Prepare the environment for training with the codebook

    Args:
        cfg (MainConfig): Main configuration object
        device (torch.device): The device to run the training on
        wandb_run (_type_, optional): Wandb object for logging. Defaults to None.
    """

    model = construct_model(cfg).to(device)

    train_dataloader, val_dataloader = get_dataloaders(cfg)
    logger.info("Validate the base model")
    base_top1_acc, base_top5_acc = validate_epoch(
        model=model, val_dataloader=val_dataloader, device=device
    )

    logger.info(
        f"Base Validation top1-accuracy {base_top1_acc}, top5-accuracy {base_top5_acc}"
    )

    # create and insert the codebook into the model, set the requires_grad
    codebook = hydra.utils.instantiate(cfg.codebook).to(device)
    if cfg.codebook_path:
        codebook.load_state_dict(torch.load(cfg.codebook_path))

    model_with_codebook = create_codebook_wrapper(
        model=model,
        codebook=codebook,
        model_name=cfg.model.name,
        unfreeze_before=cfg.training.unfreeze_before,
    )
    logger.info(f"Model with codebook: {model_with_codebook}")

    optimizers = create_optimizers(
        model=model_with_codebook,
        codebook=codebook,
        cfg=cfg,
    )
    logger.info(f"Optimizers: {optimizers}")

    epoch_iters = len(train_dataloader)
    logger.info(f"Total iterations: {epoch_iters * cfg.epochs}")
    linear_scheduler_iters = cfg.warmup_epochs * epoch_iters
    linear_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizers[0],
        start_factor=0.1,
        end_factor=1.0,
        total_iters=linear_scheduler_iters,
    )

    cosine_iterations = cfg.epochs * epoch_iters - linear_scheduler_iters
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizers[0],
        T_max=cosine_iterations,
        eta_min=0.000001,
    )
    sequential_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizers[0],
        schedulers=[linear_scheduler, cosine_scheduler],
        milestones=[linear_scheduler_iters],
    )
    logger.info(f"Warmup iterations: {linear_scheduler_iters}")
    logger.info(f"Cosine iterations: {cosine_iterations}")

    cutmix = transforms_v2.CutMix(num_classes=cfg.dataset.num_classes)
    mixup = transforms_v2.MixUp(num_classes=cfg.dataset.num_classes)
    cutmix_or_mixup = transforms_v2.RandomChoice([cutmix, mixup])
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)

    codebook_training(
        model=model_with_codebook,
        train_dataloader=train_dataloader,
        train_transforms=cutmix_or_mixup,
        val_dataloader=val_dataloader,
        optimizers=optimizers,
        schedulers=[sequential_scheduler],
        criterion=criterion,
        device=device,
        epochs=cfg.epochs,
        wandb_run=wandb_run,
    )

    if cfg.output_checkpoint_path is not None:
        hydra_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = hydra_path / f"{cfg.model.name}_{current_date}.pth"
        logger.info(f"Saving model to {out_path}")
        save_checkpoint(
            model=model,
            path=out_path,
        )
        save_checkpoint(
            model=codebook,
            path=hydra_path / f"{cfg.model.name}_codebook_{current_date}.pth",
        )


@hydra.main(config_path="config", config_name="main_config", version_base="1.2")
def start_training(cfg: MainConfig) -> None:
    """Main function for training the codebook

    Args:
        cfg (MainConfig): Hydra config object with all the settings. (Located in config/main_config.py)
    """
    logger.setLevel(cfg._logging_level)

    if cfg.wandb.is_enabled:
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg),
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            job_type=cfg.wandb.job_type,
            tags=cfg.wandb.tags,
        )
    else:
        wandb_run = None
        logger.info("wandb is not enabled")

    logger.info(OmegaConf.to_yaml(cfg))
    hydra_output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    logger.info(f"Hydra output directory: {hydra_output_dir}")
    set_reproducibility(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepare_codebook_training(cfg, device, wandb_run)
