from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import wandb
from src.models.codebook_wrappers import create_codebook_wrapper
from src.construct_model import construct_model
from src.datasets.construct_dataset import get_dataloaders, get_dataset
from src.utils import (
    CheckpointTracker,
    save_checkpoint,
    set_reproducibility,
)
from src.training import (
    train_epoch_ema_codebook,
    validate_epoch,
    validate_epoch_ema_codebook,
)
import logging
import hydra
from src.config.main_config import MainConfig
from torchvision.transforms import v2 as transforms_v2


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(config_path="config", config_name="main_config", version_base="1.2")
def start_training_ema(cfg: MainConfig) -> None:
    """Main function for training the codebook

    Args:
        cfg (PruningConfig): Configuration object containing all the parameters for the pruning process.
    """
    logger.setLevel(cfg._logging_level)

    if cfg.wandb.is_enabled:
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg),  # type: ignore
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
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir  # type: ignore
    )
    logger.info(f"Hydra output directory: {hydra_output_dir}")
    set_reproducibility(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepare_codebook_training(
        cfg=cfg,
        device=device,
        hydra_path=hydra_output_dir,
        wandb_run=wandb_run,
    )


def prepare_codebook_training(
    cfg: MainConfig, device: torch.device, hydra_path: Path, wandb_run=None
) -> None:
    """Prepare the environment for training with the codebook

    Args:
        cfg (MainConfig): Main configuration object
        device (torch.device): The device to run the training on
        wandb_run (_type_, optional): Wandb object for logging. Defaults to None.
    """

    logger.info(f"Device: {device}")
    model = construct_model(cfg, device)

    train_ds, val_ds = get_dataset(cfg)
    train_dataloader, val_dataloader = get_dataloaders(
        train_dl_config=cfg.train_dataloader,
        val_dl_config=cfg.val_dataloader,
        train_dataset=train_ds,
        val_dataset=val_ds,
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

    cutmix = transforms_v2.CutMix(num_classes=cfg.dataset.num_classes)
    mixup = transforms_v2.MixUp(num_classes=cfg.dataset.num_classes)
    cutmix_or_mixup = transforms_v2.RandomChoice([cutmix, mixup])
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)

    if cfg.training.compile_model:
        logger.info("Compiling the model for performance optimization")
        torch.set_float32_matmul_precision("high")
        model_with_codebook.compile(mode=cfg.training.compile_mode)

    codebook_training(
        model=model_with_codebook,
        cfg=cfg,
        train_dataloader=train_dataloader,
        train_transforms=cutmix_or_mixup,
        val_dataloader=val_dataloader,
        criterion=criterion,
        device=device,
        wandb_run=wandb_run,
        hydra_path=hydra_path,
    )


def codebook_training(
    model: nn.Module,
    cfg: MainConfig,
    train_dataloader: torch.utils.data.DataLoader,
    train_transforms: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    hydra_path: Path,
    wandb_run=None,
):
    # Initialize GradScaler with enabled parameter - handles AMP automatically
    scaler = torch.amp.GradScaler(device=device.type, enabled=cfg.training.use_amp)  # type: ignore
    checkpoint_tracker = CheckpointTracker()
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_name = f"{cfg.model.name}_{cfg.codebook.num_entries}_{cfg.dataset.name}_{current_date}.pth"

    if scaler.is_enabled():
        logger.info("Using Automated Mixed Precision (AMP) training")
    else:
        logger.info("Using standard precision training")

    for epoch in range(cfg.epochs):
        logger.info(f"Epoch: {epoch}")
        train_statistics = train_epoch_ema_codebook(
            model=model,
            train_dataloader=train_dataloader,
            num_classes=cfg.dataset.num_classes,
            transforms=train_transforms,
            criterion=criterion,
            task_loss_weight=cfg.training.task_loss_weight,
            codebook_loss_weight=cfg.training.codebook_loss_weight,
            device=device,
            wandb_run=wandb_run,
        )

        # validate every 5 epochs
        if epoch % 5 == 0:
            val_statistics = validate_epoch_ema_codebook(
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
            if checkpoint_tracker.is_best(accuracy):
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
