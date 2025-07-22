from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
from src.codebook_wrappers import create_codebook_wrapper
from src.construct_model import construct_model
from src.datasets.construct_dataset import get_dataloaders, get_dataset
from src.utils import (
    CheckpointTracker,
    construct_init_function,
    create_schedulers,
    validate_epoch,
    save_checkpoint,
    create_optimizers,
    create_feature_dataloader,
)
from src.training import (
    train_epoch_cosine_codebook,
    validate_epoch_cosine_codebook,
)
import logging
import hydra
from src.config.main_config import MainConfig
from torchvision.transforms import v2 as transforms_v2
from src.distributed_utils import create_samplers


logger = logging.getLogger(__name__)


def prepare_codebook_training(
    cfg: MainConfig, device: torch.device, hydra_path: Path, wandb_run=None
) -> None:
    """Prepare the environment for training with the codebook

    Args:
        cfg (MainConfig): Main configuration object
        device (torch.device): The device to run the training on
        wandb_run (_type_, optional): Wandb object for logging. Defaults to None.
    """

    local_rank = torch.distributed.get_rank()
    logger.info(f"Device: {device}")
    model = construct_model(cfg, device)

    train_ds, val_ds = get_dataset(cfg)
    train_sampler, val_sampler = create_samplers(
        train_ds,
        val_ds,
        rank=local_rank,
        world_size=cfg.distributed.world_size,
        seed=cfg.seed,
    )
    train_dataloader, val_dataloader = get_dataloaders(
        cfg, train_ds, val_ds, train_sampler=train_sampler, val_sampler=val_sampler
    )

    logger.info("Validate the base model")
    base_top1_acc, base_top5_acc = validate_epoch(
        model=model, val_dataloader=val_dataloader, device=device
    )

    logger.info(
        f"Base Validation top1-accuracy {base_top1_acc}, top5-accuracy {base_top5_acc}"
    )

    # create and insert the codebook into the model, set the requires_grad
    codebook = hydra.utils.instantiate(cfg.codebook).to(device)

    # check if codebook constains initialize_embeddings method
    if hasattr(codebook, "initialize_embeddings"):
        logger.info("Initializing codebook embeddings")
        init_function = construct_init_function(cfg.codebook_init)
        logger.info(f"Using initialization function: {init_function}")
        codebook.initialize_embeddings(init_func=init_function)

    if cfg.codebook_path:
        codebook.load_state_dict(
            torch.load(cfg.codebook_path, map_location=device, weights_only=True)
        )

    model_with_codebook = create_codebook_wrapper(
        model=model,
        codebook=codebook,
        model_name=cfg.model.name,
        unfreeze_before=cfg.training.unfreeze_before,
    )
    logger.info(f"Model with codebook: {model_with_codebook}")

    # Change to DistributedDataParallel
    model_with_codebook = nn.parallel.DistributedDataParallel(
        model_with_codebook,
        device_ids=[device.index],
        output_device=device.index,
        broadcast_buffers=True,
    )

    optimizers = create_optimizers(
        model=model_with_codebook,
        codebook=codebook,
        cfg=cfg,
    )
    logger.info(f"Optimizers: {optimizers}")

    schedulers = create_schedulers(
        optimizers=optimizers,
        epoch_iters=len(train_dataloader),
        warmup_epochs=cfg.training.warmup_epochs,
        epochs=cfg.epochs,
    )
    logger.info(f"Schedulers: {schedulers}")

    cutmix = transforms_v2.CutMix(num_classes=cfg.dataset.num_classes)
    mixup = transforms_v2.MixUp(num_classes=cfg.dataset.num_classes)
    cutmix_or_mixup = transforms_v2.RandomChoice([cutmix, mixup])
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)

    if cfg.training.only_features:
        logger.info("Extracting features from the training dataset")

        train_dataloader, val_dataloader = create_feature_dataloader(
            model=model.features,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=device,
            transforms=cutmix_or_mixup,
            local_rank=local_rank,
            cfg=cfg,
        )

        model_to_train = nn.parallel.DistributedDataParallel(
            create_codebook_wrapper(
                model=model,
                codebook=codebook,
                model_name="head_only",
                unfreeze_before=0,
            ),
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=True,
        )
    else:
        model_to_train = model_with_codebook

    if cfg.training.compile_model:
        logger.info("Compiling the model for performance optimization")
        torch.set_float32_matmul_precision("high")
        model_with_codebook.compile(mode=cfg.training.compile_mode)

    codebook_training(
        model=model_to_train,
        cfg=cfg,
        train_dataloader=train_dataloader,
        train_transforms=cutmix_or_mixup,
        val_dataloader=val_dataloader,
        optimizers=optimizers,
        schedulers=schedulers,
        criterion=criterion,
        device=device,
        wandb_run=wandb_run,
        local_rank=local_rank,
        hydra_path=hydra_path,
    )


def codebook_training(
    model: nn.Module,
    cfg: MainConfig,
    train_dataloader: torch.utils.data.DataLoader,
    train_transforms: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[torch.optim.lr_scheduler._LRScheduler],
    criterion: nn.Module,
    device: torch.device,
    local_rank: int,
    hydra_path: Path,
    wandb_run=None,
):
    # Initialize GradScaler with enabled parameter - handles AMP automatically
    scaler = torch.amp.GradScaler(device=device.type, enabled=cfg.training.use_amp) # type: ignore
    checkpoint_tracker = CheckpointTracker()
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_name = f"{cfg.model.name}_{cfg.codebook.num_entries}_{current_date}.pth"

    if scaler.is_enabled():
        logger.info("Using Automated Mixed Precision (AMP) training")
    else:
        logger.info("Using standard precision training")

    for epoch in range(cfg.epochs):
        logger.info(f"Epoch: {epoch}")
        train_statistics = train_epoch_cosine_codebook(
            model=model,
            train_dataloader=train_dataloader,
            num_classes=cfg.dataset.num_classes,
            transforms=train_transforms,
            optimizers=optimizers,
            schedulers=schedulers,
            criterion=criterion,
            task_loss_weight=cfg.training.task_loss_weight,
            codebook_loss_weight=cfg.training.codebook_loss_weight,
            device=device,
            scaler=scaler,
            wandb_run=wandb_run,
        )

        val_statistics = validate_epoch_cosine_codebook(
            model=model,
            val_dataloader=val_dataloader,
            device=device,
            num_classes=cfg.dataset.num_classes,
        )

        # add prefix Train and Validation to the statistics
        train_statistics = {f"Train {k}": v for k, v in train_statistics.items()}
        val_statistics = {f"Validation {k}": v for k, v in val_statistics.items()}

        logger.info(f"Train statistics: {train_statistics}")
        logger.info(f"Validation statistics: {val_statistics}")

        if wandb_run:
            wandb_run.log({"epoch": epoch})
            wandb_run.log(train_statistics)
            wandb_run.log(val_statistics)

        accuracy = val_statistics["Validation Top1 Accuracy"]
        if local_rank == 0 and checkpoint_tracker.is_best(accuracy):
            save_checkpoint(
                model=model,
                val_accuracy=accuracy,
                epoch=epoch,
                name=checkpoint_name,
                hydra_path=hydra_path,
                wandb_run=wandb_run,
            )

    logger.info(
        f"Best accuracy {checkpoint_tracker.best_val_accuracy}% at epoch {checkpoint_tracker.best_val_accuracy}"
    )
