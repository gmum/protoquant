import logging
from pathlib import Path
from typing import Callable
import torch
import torch.nn as nn
import random
import numpy as np
from src.distributed_utils import create_samplers
from src.training import extract_features_from_backbone
from src.models.codebook_wrappers import CNNCodebookWrapper
from src.config.main_config import MainConfig
import hydra
from src.config.codebook_init import BaseInitializationConfig
from omegaconf import OmegaConf
import functools
import uuid
from timm.scheduler import create_scheduler
from dataclasses import dataclass

logger = logging.getLogger(__name__)
UNIQUE_ID = uuid.uuid4().hex[:8]

def set_reproducibility(seed: int) -> None:
    """Set the seed for reproducibility and deterministic behavior

    Args:
        seed (int): The seed to set for reproducibility
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_optimizers(
    model: nn.Module, codebook: nn.Module, cfg: MainConfig
) -> list[torch.optim.Optimizer]:
    """Create optimizers for the model

    Args:
        model (nn.Module): The model to create optimizers for
        codebook (nn.Module): The codebook module
        cfg (MainConfig): The configuration object

    Returns:
        list[torch.optim.Optimizer]: List of optimizers
    """

    codebook.requires_grad_(False)
    base_grad_parameters = [
        param for param in model.parameters() if param.requires_grad
    ]
    codebook.requires_grad_(True)

    optimizers = []
    if base_grad_parameters:
        logger.info("Creating separate optimizer for the base model and the codebook")
        base_optimizer = hydra.utils.instantiate(
            cfg.base_optimizer, base_grad_parameters
        )
        optimizers.append(base_optimizer)

    codebook_optimizer = hydra.utils.instantiate(
        cfg.codebook_optimizer, codebook.parameters()
    )
    optimizers.append(codebook_optimizer)

    return optimizers


@dataclass
class SchedulerArgs:
    """Configuration for creating a timm scheduler."""
    sched: str = 'cosine'
    epochs: int = 100            # Fewer epochs are needed for fine-tuning
    lr: float = 5e-5             # Lower learning rate is crucial
    warmup_epochs: int = 5       # Standard value
    cooldown_epochs: int = 5     # Standard value
    min_lr: float = 1e-6         # The floor for the learning rate
    warmup_lr: float = 1e-6      # The learning rate to start the warmup from
    t_in_epochs: bool = True     # Step per-epoch via scheduler.step(epoch)


def create_schedulers(
    optimizers: list[torch.optim.Optimizer],
    scheduler_args: SchedulerArgs
) -> list[torch.optim.lr_scheduler._LRScheduler]:
    """
    Creates a DeiT-like learning rate scheduler using a configuration object.

    Args:
        optimizers (list[torch.optim.Optimizer]): The list of optimizers to wrap.
        scheduler_args (SchedulerArgs): The configuration object with all scheduler settings.

    Returns:
        list[torch.optim.lr_scheduler._LRScheduler]: A list containing the configured scheduler.
    """
    if not optimizers:
        return []

    # The timm factory function can take the dataclass instance directly.
    lr_scheduler, _ = create_scheduler(scheduler_args, optimizers[0])

    logger.info(
        f"Created '{scheduler_args.sched}' scheduler with {scheduler_args.warmup_epochs} warmup epochs "
        f"and {scheduler_args.cooldown_epochs} cooldown epochs."
    )

    return [lr_scheduler]


def calculate_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy of the model's predictions.

    Args:
        output (torch.Tensor): The output from the model.
        target (torch.Tensor): The ground truth labels.

    Returns:
        float: The accuracy.
    """

    with torch.no_grad():
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        acc = correct / target.size(0)
        return acc * 100


def construct_init_function(
    init_config: BaseInitializationConfig,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Constructs an initialization function based on the provided configuration.

    The returned function will take only one argument: the tensor to be initialized.

    Args:
        init_config (BaseInitializationConfig): Configuration for the initialization.

    Returns:
        Callable[[torch.Tensor], None]: The initialization function, ready to be called
                                       with a single tensor argument.
    """

    init_func = hydra.utils.get_method(init_config._target_)

    init_params = OmegaConf.to_container(init_config, resolve=True)
    if "_target_" in init_params:
        del init_params["_target_"]  # Remove the target, as it's already extracted

    initialized_fn = functools.partial(init_func, **init_params)

    return initialized_fn


def create_feature_dataloader(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    transforms: torch.nn.Module | None,
    local_rank: int,
    cfg: MainConfig,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create a DataLoader for feature extraction.

    Args:
        model (nn.Module): The model to extract features from.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        device (torch.device): Device to run the model on ('cuda' or 'cpu').
        transforms (torch.nn.Module): Transforms to apply to the input data.
        local_rank (int): The rank of the current process in distributed training.
        cfg (MainConfig): Main configuration object.

    Returns:
        tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: A tuple containing feature DataLoaders for training and validation.
    """
    # extract features into the dataset

    train_features, train_labels = extract_features_from_backbone(
        feature_backbone=model,
        dataloader=train_dataloader,
        device=device,
        transforms=transforms,
    )
    val_features, val_labels = extract_features_from_backbone(
        feature_backbone=model,
        dataloader=val_dataloader,
        device=device,
        transforms=None,
    )
    train_features_ds = torch.utils.data.TensorDataset(train_features, train_labels)
    val_features_ds = torch.utils.data.TensorDataset(val_features, val_labels)

    train_sampler, val_sampler = create_samplers(
        train_dataset=train_features_ds,
        val_dataset=val_features_ds,
        rank=local_rank,
        world_size=cfg.distributed.world_size,
        seed=cfg.seed,
    )

    train_dl = torch.utils.data.DataLoader(
        train_features_ds,
        batch_size=cfg.train_dataloader.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=cfg.train_dataloader.num_workers,
        pin_memory=cfg.train_dataloader.pin_memory,
        persistent_workers=True,
    )

    val_dl = torch.utils.data.DataLoader(
        val_features_ds,
        batch_size=cfg.val_dataloader.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.val_dataloader.num_workers,
        pin_memory=cfg.val_dataloader.pin_memory,
        persistent_workers=True,
    )

    return train_dl, val_dl


class CheckpointTracker:
    """Tracks the best validation accuracy and saves checkpoints accordingly"""

    def __init__(self, initial_best: float = 0.0):
        self.best_val_accuracy = initial_best

    def is_best(self, val_accuracy: float) -> bool:
        """Check if the current validation accuracy is the best so far

        Args:
            val_accuracy (float): The validation accuracy to check

        Returns:
            bool: True if this is the best validation accuracy, False otherwise
        """

        if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            return True

        return False


def save_checkpoint(
    model: nn.parallel.DistributedDataParallel | nn.Module | CNNCodebookWrapper,
    val_accuracy: float,
    epoch: int,
    name: str,
    hydra_path: Path,
    wandb_run=None,
    save_wandb: bool = False,
) -> bool:
    """Save model and codebook checkpoint

    Args:
        model (nn.Module): The model to save (can be DDP wrapped)
        val_accuracy (float): The validation accuracy achieved
        cfg (MainConfig): Configuration object
        hydra_path (Path): Path to save checkpoints
        wandb_run: WandB run object for uploading checkpoints

    Returns:
        bool: True if checkpoint was saved successfully
    """

    base_model = model
    if isinstance(model, nn.parallel.DistributedDataParallel):
        base_model = model.module

    if hasattr(base_model, "codebook"):
        codebook = base_model.codebook
    else:
        raise ValueError("Model does not have a codebook attribute")

    # Save model
    model_path = hydra_path / f"model_{name}_{UNIQUE_ID}.pth"

    # Save codebook
    codebook_path = hydra_path / f"codebook_{name}_{UNIQUE_ID}.pth"
    logger.info(f"Saving codebook to {codebook_path}")
    torch.save(codebook.state_dict(), codebook_path)

    # Save to wandb if available
    if wandb_run:
        # Log checkpoint info
        wandb_run.log(
            {
                "Checkpoint_Val_Accuracy": val_accuracy,
                "Checkpoint_Epoch": epoch,
            }
        )

        if save_wandb:
            wandb_run.save(
                str(model_path),
                base_path=hydra_path,
                policy="now",
            )

            wandb_run.save(
                str(codebook_path),
                base_path=hydra_path,
                policy="now",
            )

    logger.info(
        f"Checkpoint saved. Validation accuracy: {val_accuracy:.2f}% at epoch {epoch}"
    )
    return True
