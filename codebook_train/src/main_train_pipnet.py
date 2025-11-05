import logging
from pathlib import Path
from datetime import datetime
import uuid

from src.pipnet_utils import TrainingWrapper, build_pipnet_model
from src.datasets.transforms import get_default_image_transforms, get_deit_transforms
import torch
import torch.nn as nn
import torch.distributed as dist
import hydra
import wandb
from omegaconf import OmegaConf
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy

from src.config.pipnet_config import PipNetConfig
from src.datasets.construct_dataset import get_dataset, get_dataloaders
from src.distributed_utils import create_samplers, setup_hydra_logging_for_rank
from src.utils import (
    SchedulerArgs,
    set_reproducibility,
    create_schedulers,
)
from src.training import train_epoch, validate_epoch
from src.training import train_epoch_pipnet  # add specialized epoch function


logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="pipnet_config", version_base="1.2")
def start_training_pipnet(cfg: PipNetConfig) -> None:
    """
    Start distributed training for PIPNet head with loaded codes.
    Reuses DDP setup and logging pattern from main.
    """
    main_logger = logging.getLogger(__name__)
    main_logger.setLevel(cfg._logging_level)

    main_logger.info(OmegaConf.to_yaml(cfg))
    hydra_output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir # type: ignore
    ) 
    main_logger.info(f"Hydra output directory: {hydra_output_dir}")

    main_logger.info(
        f"Starting distributed training with {cfg.distributed.world_size} processes using {cfg.distributed.backend} backend."
    )

    torch.multiprocessing.spawn(  # type: ignore
        fn=_distributed_worker,
        args=(cfg, hydra_output_dir),
        nprocs=cfg.distributed.world_size,
        join=True,
    )


def _distributed_worker(rank: int, cfg: PipNetConfig, hydra_output_dir: Path) -> None:
    setup_hydra_logging_for_rank(rank, cfg._logging_level, hydra_output_dir)

    if rank == 0 and cfg.wandb.is_enabled:
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            job_type=cfg.wandb.job_type,
            tags=cfg.wandb.tags,
        )
    else:
        wandb_run = None

    dist.init_process_group(
        backend=cfg.distributed.backend,
        init_method=cfg.distributed.init_method,
        world_size=cfg.distributed.world_size,
        rank=rank,
    )
    device = torch.device(f"cuda:{rank}")

    # Reproducibility per-rank
    set_reproducibility(cfg.seed + rank)

    try:
        prepare_and_train(cfg, device, hydra_output_dir, wandb_run)
    finally:
        if wandb_run and rank == 0:
            wandb_run.finish()
        dist.destroy_process_group()


def prepare_and_train(
    cfg: PipNetConfig, device: torch.device, hydra_path: Path, wandb_run=None
) -> None:
    local_rank = dist.get_rank()
    logger.info(f"[Rank {local_rank}] Device: {device}")

    # Data
    if cfg.dataset.use_deit_transforms:
        train_transform, val_transform = get_deit_transforms(is_precropped=(cfg.dataset.name == "cub200"))
    else:
        train_transform, val_transform = get_default_image_transforms(
            autoaugment=cfg.dataset.autoaugment,
            resize_value=224 if cfg.dataset.name == "cub200" else 256,
            crop_value=None if cfg.dataset.name == "cub200" else 224,
            random_erase=cfg.dataset.random_erase,
            horizontal_flip=cfg.dataset.horizontal_flip,
            is_precropped=cfg.dataset.name == "cub200",
        )

    train_ds, val_ds = get_dataset(
        name=cfg.dataset.name,
        train_transform=train_transform,
        val_transform=val_transform,
        path=cfg.dataset._path,
    )
    train_sampler, val_sampler = create_samplers(
        train_ds,
        val_ds,
        rank=local_rank,
        world_size=cfg.distributed.world_size,
        seed=cfg.seed,
    )
    train_dl, val_dl = get_dataloaders(
        train_dl_config=cfg.train_dataloader,
        val_dl_config=cfg.val_dataloader,
        train_dataset=train_ds,
        val_dataset=val_ds,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
    )

    # Build model and head
    model, head = build_pipnet_model(cfg, device)
    model = TrainingWrapper(model)
    
    # log the size of the codebook
    num_codes = head.codebook.shape[0]
    logger.info(f"Codebook size: {num_codes}")
    if wandb_run:
        wandb_run.config.update({"codebook_size": num_codes})

    # Validate base backbone (classifier head is ignored; we validate with PIPNetHead)
    logger.info("Quick sanity validation (random-initialized PIPNet head)")
    base_top1, base_top5 = validate_epoch(
        model=model, val_dataloader=val_dl, device=device
    )
    logger.info(f"Sanity Validation - Top1: {base_top1:.2f}%, Top5: {base_top5:.2f}%")

    # Wrap in DDP
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device.index],
        output_device=device.index,
        broadcast_buffers=True,
        find_unused_parameters=True,
    )
    logger.info(f"Model: {model}")

    # Optimizer: train only the PIPNet head
    # Reuse base optimizer config for the head
    head_optimizer = hydra.utils.instantiate(cfg.base_optimizer, head.parameters())

    # LR schedulers
    scheduler_args = SchedulerArgs(
        epochs=cfg.epochs,
        warmup_epochs=cfg.training.warmup_epochs,
        lr=cfg.base_optimizer.lr
    )
    scheduler = create_schedulers(
        optimizers=[head_optimizer],
        scheduler_args=scheduler_args
    )[0]
    logger.info(f"Scheduler: {scheduler}")

    # Criterion and transforms
    mixup_fn = Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        cutmix_minmax=None,
        prob=1.0,
        switch_prob=0.5,
        mode='batch',
        label_smoothing=cfg.training.label_smoothing,
        num_classes=cfg.dataset.num_classes
    )
    criterion = SoftTargetCrossEntropy()

    # Train
    train_loop(
        model=model,
        cfg=cfg,
        train_dataloader=train_dl,
        train_transforms=mixup_fn, # type: ignore
        val_dataloader=val_dl,
        optimizer=head_optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        local_rank=local_rank,
        hydra_path=hydra_path,
        wandb_run=wandb_run,
    )


def train_loop(
    model: nn.Module,
    cfg: PipNetConfig,
    train_dataloader: torch.utils.data.DataLoader,
    train_transforms: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    device: torch.device,
    local_rank: int,
    hydra_path: Path,
    wandb_run=None,
) -> None:
    best_top1 = 0.0
    run_uuid = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    best_ckpt_path = hydra_path / f"pipnet_{cfg.model.name}_{timestamp}_{run_uuid}.pth"

    # Regularization weight from config (default 0.0 if absent)
    reg_weight = cfg.training.pipnet_regularization_weight

    for epoch in range(cfg.epochs):
        logger.info(f"Epoch: {epoch}")

        # Ensure DistributedSampler shuffling
        if hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)  # type: ignore[attr-defined]

        # Run one epoch with PIPNet regularization
        train_epoch_pipnet(
            model=model,
            train_dataloader=train_dataloader,
            transforms=train_transforms,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            reg_weight=reg_weight,
            wandb_run=wandb_run,
        )

        # Validation using existing helper (expects model(inputs) -> logits)
        top1, top5 = validate_epoch(
            model=model, val_dataloader=val_dataloader, device=device
        )

        scheduler.step(epoch)

        val_logs = {"Validation Top1 Accuracy": top1, "Validation Top5 Accuracy": top5}
        logger.info(val_logs)
        if wandb_run:
            wandb_run.log({"epoch": epoch, **val_logs})

        # Save best on rank 0 (overwrite the same file name)
        if local_rank == 0 and top1 > best_top1:
            best_top1 = top1
            saved_path = save_pipnet_checkpoint(model, best_ckpt_path, epoch, top1)
            logger.info(f"Saved best checkpoint to: {saved_path}")

    logger.info(f"Best Top1 Accuracy: {best_top1:.2f}%")


def save_pipnet_checkpoint(
    model: nn.parallel.DistributedDataParallel | nn.Module,
    ckpt_path: Path,
    epoch: int,
    val_top1: float,
) -> Path:
    """
    Save only the best checkpoint to a fixed path (overwrites on improvement).
    File name should already include the 'pipnet_{model}_{timestamp}_{uuid}.pth' pattern.
    """
    base = (
        model.module
        if isinstance(model, nn.parallel.DistributedDataParallel)
        else model
    )
    payload = {
        "model": base.model_to_wrap.state_dict(),
        "epoch": epoch,
        "val_top1": val_top1,
    }
    torch.save(payload, ckpt_path)
    return ckpt_path


if __name__ == "__main__":
    start_training_pipnet()  # type: ignore
