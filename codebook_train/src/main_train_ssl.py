import hydra
import logging
from omegaconf import OmegaConf
from src.models.codebook_wrappers import create_codebook_wrapper
from src.datasets.construct_dataset import get_dataloaders
from src.construct_model import construct_model
from src.utils import set_reproducibility, save_checkpoint, construct_init_function
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from src.config.ssl_config import SelfSupervisedConfig
from torchvision.transforms import v2 as transforms_v2
from src.ssl_utils import evaluate_linear_probe, train_epoch_ssl, create_scheduler
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="ssl_config", version_base="1.2")
def start_training_ssl(cfg: SelfSupervisedConfig) -> None:
    """Main function for training the codebook with self-supervised learning

    Args:
        cfg (SelfSupervisedConfig): Main configuration object
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
    train_ssl_training(cfg, device, wandb_run)


def train_ssl_training(
    cfg: SelfSupervisedConfig, device: torch.device, wandb_run=None
) -> None:
    """Trains the codebook with self-supervised learning

    Args:
        cfg (SelfSupervisedConfig): Main configuration object
        device (torch.device): Device to use for training
        wandb_run: wandb run object
    """

    model = construct_model(cfg, device=device)
    logger.info(f"Model: {model}")

    train_dataloader, val_dataloader = get_dataloaders(cfg)

    linear_probe = nn.Linear(cfg.training.probe_dim, cfg.dataset.num_classes).to(device)
    logger.info(f"Linear probe: {linear_probe}")

    probe_optimizer = hydra.utils.instantiate(
        cfg.probe_optimizer,
        params=linear_probe.parameters(),
    )
    logger.info(f"Linear probe optimizer: {probe_optimizer}")

    cutmix = transforms_v2.CutMix(num_classes=cfg.dataset.num_classes)
    mixup = transforms_v2.MixUp(num_classes=cfg.dataset.num_classes)
    cutmix_or_mixup = transforms_v2.RandomChoice([cutmix, mixup])
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)

    epoch_iters = len(train_dataloader)
    probe_scheduler = None
    if cfg.training.enable_schedulers:
        probe_scheduler = create_scheduler(
            probe_optimizer,
            epochs=cfg.training.probe_epochs,
            epoch_iters=epoch_iters,
            warmup_epochs=cfg.training.warmup_epochs,
        )
    logger.info("Validate the base model")

    # freeze the model
    for param in model.parameters():
        param.requires_grad = False

    linear_probe_acc = evaluate_linear_probe(
        model=model,
        train_dl=train_dataloader,
        train_transforms=cutmix_or_mixup,
        val_dl=val_dataloader,
        linear_probe=linear_probe,
        optimizer=probe_optimizer,
        scheduler=probe_scheduler,
        epochs=cfg.training.probe_epochs,
        criterion=criterion,
        apply_adaptive_pooling=True,
        device=device,
    )
    logger.info(f"Base linear probe accuracy: {linear_probe_acc:.4f}")

    # create and insert the codebook into the model, set the requires_grad
    codebook = hydra.utils.instantiate(cfg.codebook).to(device)
    if cfg.codebook_path:
        codebook.load_state_dict(torch.load(cfg.codebook_path))

    # check if codebook constains initialize_embeddings method
    if hasattr(codebook, "initialize_embeddings"):
        logger.info("Initializing codebook embeddings")
        init_function = construct_init_function(cfg.codebook_init)
        logger.info(f"Using initialization function: {init_function}")
        codebook.initialize_embeddings(init_func=init_function)

    model_with_codebook = create_codebook_wrapper(
        model=model,
        codebook=codebook,
        model_name=cfg.model.name,
        unfreeze_before=0,
    )
    logger.info(f"Model with codebook: {model_with_codebook}")

    codebook_optimizer = hydra.utils.instantiate(
        cfg.codebook_optimizer,
        params=codebook.parameters(),
    )
    logger.info(f"Codebook optimizer: {codebook_optimizer}")

    codebook_scheduler = None
    if cfg.training.enable_schedulers:
        codebook_scheduler = create_scheduler(
            codebook_optimizer,
            epochs=cfg.epochs,
            epoch_iters=epoch_iters,
            warmup_epochs=cfg.training.warmup_epochs,
        )
    logger.info(f"Codebook scheduler: {codebook_scheduler}")

    # Reset the scheduler for the probe optimizer
    if cfg.training.enable_schedulers and probe_scheduler is not None:
        probe_scheduler = create_scheduler(
            probe_optimizer,
            epochs=cfg.training.probe_epochs,
            epoch_iters=epoch_iters,
            warmup_epochs=cfg.training.warmup_epochs,
        )

    train_codebook_ssl(
        model=model_with_codebook,
        epochs=cfg.epochs,
        train_dataloader=train_dataloader,
        train_transforms=cutmix_or_mixup,
        val_dataloader=val_dataloader,
        codebook_optimizer=codebook_optimizer,
        probe_optimizer=probe_optimizer,
        linear_probe=linear_probe,
        probe_criterion=criterion,
        probe_epochs=cfg.training.probe_epochs,
        codebook_scheduler=codebook_scheduler,
        probe_scheduler=probe_scheduler,
        device=device,
        wandb_run=wandb_run,
    )

    if cfg.output_checkpoint_path is not None:
        hydra_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = hydra_path / f"{cfg.model.name}_ssl_{current_date}.pth"
        logger.info(f"Saving model to {out_path}")
        save_checkpoint(
            model=model,
            path=out_path,
        )

        codebook_path = hydra_path / f"{cfg.model.name}_codebook_ssl_{current_date}.pth"
        save_checkpoint(
            model=codebook,
            path=codebook_path,
        )

        # save to wandb if available
        if wandb_run:
            wandb.save(
                str(out_path),
                base_path=hydra_path,
                policy="now",
            )

            # save checkpoint for codebook
            wandb.save(
                str(codebook_path),
                base_path=hydra_path,
                policy="now",
            )


def train_codebook_ssl(
    model: nn.Module,
    epochs: int,
    train_dataloader: torch.utils.data.DataLoader,
    train_transforms: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    codebook_optimizer: torch.optim.Optimizer,
    probe_optimizer: torch.optim.Optimizer,
    linear_probe: nn.Module,
    probe_criterion: nn.Module,
    probe_epochs: int,
    codebook_scheduler: torch.optim.lr_scheduler._LRScheduler,
    probe_scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    wandb_run=None,
):
    for epoch in range(epochs):
        logger.info(f"Epoch: {epoch}")
        train_statistics = train_epoch_ssl(
            model=model,
            train_dataloader=train_dataloader,
            transforms=train_transforms,
            optimizers=[codebook_optimizer],
            schedulers=[codebook_scheduler] if codebook_scheduler else [],
            device=device,
        )

        # add prefix Train and Validation to the statistics
        train_statistics = {f"Train {k}": v for k, v in train_statistics.items()}

        logger.info(f"Train statistics: {train_statistics}")
        if wandb_run:
            wandb.log(
                {
                    "Epoch": epoch,
                    **train_statistics,
                }
            )

    logger.info("Linear Probe Evaluation")
    probe_acc = evaluate_linear_probe(
        model=model,
        train_dl=train_dataloader,
        train_transforms=train_transforms,
        val_dl=val_dataloader,
        linear_probe=linear_probe,
        optimizer=probe_optimizer,
        epochs=probe_epochs,
        criterion=probe_criterion,
        scheduler=probe_scheduler,
        apply_adaptive_pooling=True,
        device=device,
    )

    logger.info(f"Linear Probe Accuracy: {probe_acc:.4f}%")
    if wandb_run:
        wandb.log(
            {
                "Linear Probe Accuracy": probe_acc,
            }
        )
