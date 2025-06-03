import torch.distributed as dist
import logging
from typing import Any
import torch
import torch.nn as nn
from src.codebook import CosineSimilarityCodebook
from torchmetrics import Accuracy

logger = logging.getLogger(__name__)


def train_epoch_cosine_codebook(
    model: nn.parallel.DistributedDataParallel,
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
        model (nn.parallel.DistributedDataParallel): DDP wrapped model with codebook
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
    log_interval = len(train_dataloader) // 5

    accuracy_metric = Accuracy(task="multiclass", num_classes=200).to(device)

    for batch, (images, labels) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        transformed_images, transformed_labels = transforms(images, labels)

        for optimizer in optimizers:
            optimizer.zero_grad()

        # Forward pass with autocast
        with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            logits, codebook_loss = model(transformed_images)
            task_loss = criterion(logits, transformed_labels)
            total_loss = (
                task_loss_weight * task_loss + codebook_loss_weight * codebook_loss
            )

        # Backward pass with scaling
        scaler.scale(total_loss).backward()

        # Unscale gradients and step optimizers
        for optimizer in optimizers:
            scaler.step(optimizer)
        scaler.update()

        # Step schedulers after each batch
        for scheduler in schedulers:
            scheduler.step()

        should_log = (batch + 1) % log_interval == 0
        if should_log:
            accuracy_metric.update(logits, labels)
            accuracy = accuracy_metric.compute() * 100
            accuracy_metric.reset()

            log_dict = {
                "Total Loss": total_loss.item(),
                "Task Loss": task_loss.item(),
                "Codebook Loss": codebook_loss.item(),
                "Top1 Accuracy": accuracy.item(),
            }

            # Only add scaler scale if AMP is enabled
            log_dict.update(
                {"Scaler Scale": scaler.get_scale()} if scaler.is_enabled() else {}
            )

            if wandb_run:
                wandb_run.log(log_dict)

            logger.info(f"Batch: {batch + 1} / {len(train_dataloader)}")
            logger.info(log_dict)

    codebook: CosineSimilarityCodebook = model.module.codebook
    codebook_statistics = codebook.get_statistics()
    codebook.reset_statistics()

    return codebook_statistics


def validate_epoch_cosine_codebook(
    model: nn.parallel.DistributedDataParallel,
    val_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Validate the model for a single epoch"""

    model.eval()

    # Initialize torchmetrics for validation
    top1_accuracy = Accuracy(task="multiclass", num_classes=200, top_k=1).to(device)
    top5_accuracy = Accuracy(task="multiclass", num_classes=200, top_k=5).to(device)

    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )

            logits, _ = model(inputs)

            # Update metrics
            top1_accuracy.update(logits, labels)
            top5_accuracy.update(logits, labels)

    # Compute final accuracies (automatically handles distributed reduction)
    top1_acc = top1_accuracy.compute() * 100
    top5_acc = top5_accuracy.compute() * 100

    codebook: CosineSimilarityCodebook = model.module.codebook
    codebook_statistics = codebook.get_statistics()
    codebook.reset_statistics()

    # Convert to Python scalars only for logging/return
    codebook_statistics["Top1 Accuracy"] = top1_acc.item()
    codebook_statistics["Top5 Accuracy"] = top5_acc.item()

    logger.info(
        f"Validation - Top1: {top1_acc.item():.2f}%, Top5: {top5_acc.item():.2f}%"
    )

    return codebook_statistics
