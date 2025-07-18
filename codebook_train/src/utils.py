import logging
from pathlib import Path
from typing import Any, Callable
import torch
import torch.nn as nn
import random
import numpy as np
from src.distributed_utils import create_samplers
from src.training import extract_features_from_backbone
from src.codebook_wrappers import CNNCodebookWrapper
from src.config.main_config import MainConfig
import hydra
from src.config.codebook_init import BaseInitializationConfig
from omegaconf import OmegaConf
import functools

logger = logging.getLogger(__name__)


def validate_epoch(
    model: nn.Module, val_dataloader: torch.utils.data.DataLoader, device: torch.device
) -> tuple[float, float]:
    model.eval()
    top1_correct, top5_correct, total = 0, 0, 0

    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            logits = model(inputs)
            _, pred = logits.topk(5, 1, True, True)

            total += labels.size(0)
            top1_correct += (pred[:, :1] == labels.view(-1, 1)).sum().item()
            top5_correct += (pred == labels.view(-1, 1)).sum().item()

    top1_acc = (top1_correct / total) * 100
    top5_acc = (top5_correct / total) * 100
    return top1_acc, top5_acc


def train_epoch(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    transforms: torch.nn.Module,
    optimizers: list[torch.optim.Optimizer],
    criterion: nn.Module,
    device: torch.device,
    wandb_run=None,
) -> None:
    model.train()

    for i, (images, labels) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        transformed_images, transformed_labels = transforms(images, labels)

        for optimizer in optimizers:
            optimizer.zero_grad()

        logits = model(transformed_images)
        loss = criterion(logits, transformed_labels)
        loss.backward()

        for optimizer in optimizers:
            optimizer.step()

        accuracy = (logits.argmax(1) == labels).float().mean()

        if wandb_run:
            wandb_run.log(
                {"Train Loss": loss.item(), "Train Accuracy": accuracy.item()}
            )

        if i % (len(train_dataloader) // 10) == 0:
            logger.info(
                f"Iteration: {i} / {len(train_dataloader)}, Loss: {loss.item()}, Accuracy: {accuracy.item()}"
            )


def train_epoch_cosine_codebook(
    model: CNNCodebookWrapper,
    train_dataloader: torch.utils.data.DataLoader,
    transforms: torch.nn.Module,
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[torch.optim.lr_scheduler._LRScheduler],
    criterion: nn.Module,
    task_loss_weight: float,
    codebook_loss_weight: float,
    device: torch.device,
    scaler: torch.amp.GradScaler = None,
    wandb_run=None,
) -> dict[str, Any]:
    """Train a single epoch of the model with cosine codebook

    Args:
        model (CNNCodebookWrapper): Wrapper around ConvNext model with cosine codebook
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data
        transforms (torch.nn.Module): Transformations to apply to the input data
        optimizers (list[torch.optim.Optimizer]): List of optimizers to use
        schedulers (list[torch.optim.lr_scheduler._LRScheduler]): List of schedulers to use
        criterion (nn.Module): Loss function to use
        task_loss_weight (float): Weight for the task loss
        codebook_loss_weight (float): Weight for the codebook loss
        device (torch.device): Device to use for training
        scaler (torch.amp.GradScaler): GradScaler for mixed precision training
        wandb_run (_type_, optional): Wandb object for logging. Defaults to None.

    Returns:
        dict[str, float]: Statistics of the codebook after training
    """

    model.train()

    for batch, (images, labels) in enumerate(train_dataloader):
        images, labels = (
            images.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )
        transformed_images, transformed_labels = transforms(images, labels)

        for optimizer in optimizers:
            optimizer.zero_grad()

        # Forward pass with autocast - enabled parameter handles AMP automatically
        with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            logits, codebook_loss = model(transformed_images)
            task_loss = criterion(logits, transformed_labels)
            total_loss = (
                task_loss_weight * task_loss + codebook_loss_weight * codebook_loss
            )

        # Backward pass with scaling - scaler handles whether to scale or not
        scaler.scale(total_loss).backward()

        # Unscale gradients and step optimizers
        for optimizer in optimizers:
            scaler.step(optimizer)
        scaler.update()

        # Step schedulers after each epoch
        for scheduler in schedulers:
            scheduler.step()

        if (batch + 1) % (len(train_dataloader) // 5) == 0:
            accuracy = (logits.argmax(1) == labels).float().mean()
            log_dict = {
                "Total Loss": total_loss.item(),
                "Task Loss": task_loss.item(),
                "Codebook Loss": codebook_loss.item(),
                "Top1 Accuracy": accuracy.item() * 100,
            }

            # Only log scaler scale if AMP is enabled
            if scaler.is_enabled():
                log_dict["Scaler Scale"] = scaler.get_scale()

            if wandb_run:
                wandb_run.log(log_dict)

            logger.info(f"Batch: {batch + 1} / {len(train_dataloader)}")
            logger.info(log_dict)

    codebook_statistics = model.codebook.get_statistics()
    model.codebook.reset_statistics()

    return codebook_statistics


def validate_epoch_cosine_codebook(
    model: nn.Module, val_dataloader: torch.utils.data.DataLoader, device: torch.device
) -> dict[str, Any]:
    """Validate the model for a single epoch

    Args:
        model (nn.Module): The model to validate
        val_dataloader (torch.utils.data.DataLoader): DataLoader for validation data
        device (torch.device): Device to use for validation

    Returns:
        dict[str, Any]: Statistics of the codebook after validation
    """

    model.eval()
    top1_correct, top5_correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            logits, _ = model(inputs)
            _, pred = logits.topk(5, 1, True, True)

            total += labels.size(0)
            top1_correct += (pred[:, :1] == labels.view(-1, 1)).sum().item()
            top5_correct += (pred == labels.view(-1, 1)).sum().item()

    top1_acc = (top1_correct / total) * 100
    top5_acc = (top5_correct / total) * 100

    codebook_statistics = model.codebook.get_statistics()
    model.codebook.reset_statistics()
    codebook_statistics["Top1 Accuracy"] = top1_acc
    codebook_statistics["Top5 Accuracy"] = top5_acc

    return codebook_statistics


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


def create_schedulers(
    optimizers: list[torch.optim.Optimizer], cfg: MainConfig, epoch_iters: int
) -> list[torch.optim.lr_scheduler._LRScheduler]:
    """Create schedulers for the optimizers

    Args:
        optimizers (list[torch.optim.Optimizer]): List of optimizers
        cfg (MainConfig): The configuration object
        epoch_iters (int): Number of iterations per epoch

    Returns:
        list[torch.optim.lr_scheduler._LRScheduler]: List of schedulers
    """

    if not cfg.training.enable_schedulers:
        logger.info("Schedulers are disabled")
        return []

    schedulers: list[torch.optim.lr_scheduler._LRScheduler] = []
    total_iterations = epoch_iters * cfg.epochs
    logger.info(f"Total iterations: {total_iterations}")

    linear_scheduler_iters = cfg.training.warmup_epochs * epoch_iters
    linear_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizers[0],
        start_factor=0.1,
        end_factor=1.0,
        total_iters=linear_scheduler_iters,
    )

    cosine_cycles = 3
    remaining_iterations = total_iterations - linear_scheduler_iters
    initial_cycle_length = remaining_iterations // (2**cosine_cycles - 1)

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizers[0],
        T_0=initial_cycle_length,
        T_mult=2,
        eta_min=0.000001,
    )
    sequential_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizers[0],
        schedulers=[linear_scheduler, cosine_scheduler],
        milestones=[linear_scheduler_iters],
    )

    schedulers.append(sequential_scheduler)
    logger.info(f"Warmup iterations: {linear_scheduler_iters}")
    logger.info(
        f"Cosine iterations: {[initial_cycle_length * 2**i for i in range(1, cosine_cycles + 1)]}"
    )

    return schedulers


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
    transforms: torch.nn.Module,
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
    model_path = hydra_path / f"model_{name}.pth"
    logger.info(
        f"Saving model (val_acc={val_accuracy:.2f}% at epoch {epoch}) to {model_path}"
    )
    torch.save(base_model.state_dict(), model_path)

    # Save codebook
    codebook_path = hydra_path / f"codebook_{name}.pth"
    logger.info(f"Saving codebook to {codebook_path}")
    torch.save(codebook.state_dict(), codebook_path)

    # Save to wandb if available
    if wandb_run:
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

        # Log checkpoint info
        wandb_run.log(
            {
                "Checkpoint_Val_Accuracy": val_accuracy,
                "Checkpoint_Epoch": epoch,
            }
        )

    logger.info(
        f"Checkpoint saved. Validation accuracy: {val_accuracy:.2f}% at epoch {epoch}"
    )
    return True
