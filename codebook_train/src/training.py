import logging
from typing import Any
import torch
import torch.nn as nn
from src.models.codebook_wrappers import CNNCodebookWrapper
from src.codebook import CosineSimilarityCodebook
from torchmetrics import Accuracy

logger = logging.getLogger(__name__)


def train_epoch_cosine_codebook(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    num_classes: int,
    transforms: torch.nn.Module,
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[torch.optim.lr_scheduler._LRScheduler],
    criterion: nn.Module,
    task_loss_weight: float,
    codebook_loss_weight: float,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    wandb_run=None,
) -> dict[str, torch.Tensor | float]:
    """Train a single epoch of the model with cosine codebook
    Args:
        model (nn.Module): Model with a codebook
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data
        num_classes (int): Number of classes in the dataset
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
        dict[str, torch.Tensor | float]: Dictionary with training statistics
    """

    model.train()
    log_interval = (len(train_dataloader) // 5) or 1

    accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)

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

    if isinstance(model, nn.parallel.DistributedDataParallel):
        codebook: CosineSimilarityCodebook = model.module.codebook
    elif isinstance(model, CNNCodebookWrapper):
        codebook: CosineSimilarityCodebook = model.codebook # type: ignore
    else:
        codebook: CosineSimilarityCodebook = model # type: ignore

    codebook_statistics = codebook.get_statistics()
    codebook.reset_statistics()

    return codebook_statistics


def validate_epoch_cosine_codebook(
    model: nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    num_classes: int,
    device: torch.device,
) -> dict[str, Any]:
    """Validate the model for a single epoch"""

    model.eval()

    # Initialize torchmetrics for validation
    top1_accuracy = Accuracy(task="multiclass", num_classes=num_classes, top_k=1).to(
        device
    )
    top5_accuracy = Accuracy(task="multiclass", num_classes=num_classes, top_k=5).to(
        device
    )

    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = (
                inputs.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )

            logits, _ = model(inputs)

            # Update metrics
            top1_accuracy.update(logits, labels)
            top5_accuracy.update(logits, labels)

    # Compute final accuracies (automatically handles distributed reduction)
    top1_acc = top1_accuracy.compute() * 100
    top5_acc = top5_accuracy.compute() * 100

    if isinstance(model, nn.parallel.DistributedDataParallel):
        codebook = model.module.codebook
    elif isinstance(model, CNNCodebookWrapper):
        codebook = model.codebook
    else:
        codebook = model

    # Convert to Python scalars only for logging/return
    codebook_statistics = codebook.get_statistics()
    codebook_statistics["Top1 Accuracy"] = top1_acc.item()
    codebook_statistics["Top5 Accuracy"] = top5_acc.item()

    logger.info(
        f"Validation - Top1: {top1_acc.item():.2f}%, Top5: {top5_acc.item():.2f}%"
    )

    return codebook_statistics


def extract_features_from_backbone(
    feature_backbone: nn.Module | nn.parallel.DistributedDataParallel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    transforms: torch.nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract features from a model using the provided dataloader

    Args:
        feature_backbone (nn.Module): The backbone model to extract features from
        dataloader (DataLoader): DataLoader containing the data
        device (torch.device): Device to run extraction on
        transforms (torch.nn.Module): Transforms to apply (only for training data)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Features and labels tensors with preserved dimensions
    """

    feature_backbone.eval()
    log_interval = len(dataloader) // 10
    all_features = []
    all_labels = []

    logger.info("Starting feature extraction")

    with torch.no_grad():
        for batch, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            if transforms is not None:
                transformed_images, _ = transforms(images, labels)
            else:
                transformed_images, _ = images, labels

            features = feature_backbone(transformed_images)
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

            should_log = ((batch + 1) % log_interval == 0) or (
                batch == len(dataloader) - 1
            )
            if should_log:
                logger.info(f"Batch: {batch + 1} / {len(dataloader)}")

    logger.info(
        f"Extracted {len(all_features)} batches with feature shape {list(all_features[0].shape)}"
    )
    # Concatenate all batches - preserving feature dimensions
    features_tensor = torch.cat(all_features, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    logger.info(
        f"Extracted {len(features_tensor)} samples with feature shape {list(features_tensor.shape)}"
    )

    # print memory size of features and labels tensors
    logger.info(
        f"Features tensor size: {features_tensor.element_size() * features_tensor.nelement() / (1024**2):.2f} MB"
    )

    return features_tensor, labels_tensor


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
    schedulers: list[torch.optim.lr_scheduler._LRScheduler] | None = None,
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

        if schedulers is not None:
            for scheduler in schedulers:
                scheduler.step()

        accuracy = (logits.argmax(1) == labels).float().mean()

        if wandb_run:
            wandb_run.log(
                {"Train Loss": loss.item(), "Train Accuracy": accuracy.item()}
            )

        if i % ((len(train_dataloader) // 5) or 1) == 0:
            logger.info(
                f"Iteration: {i} / {len(train_dataloader)}, Loss: {loss.item()}, Accuracy: {accuracy.item()}"
            )


def train_epoch_ema_codebook(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    num_classes: int,
    transforms: torch.nn.Module,
    criterion: nn.Module,
    task_loss_weight: float,
    codebook_loss_weight: float,
    device: torch.device,
    wandb_run=None,
) -> dict[str, torch.Tensor | float]:
    """Train a single epoch of the model with an EMA codebook.
    The codebook is updated via EMA during the forward pass.
    The codebook loss is used to train the encoder.
    """

    model.train()
    log_interval = len(train_dataloader) // 5

    accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    for batch, (images, labels) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        transformed_images, transformed_labels = transforms(images, labels)

        # Forward pass
        logits, codebook_loss = model(transformed_images)
        task_loss = criterion(logits, transformed_labels)
        total_loss = task_loss_weight * task_loss + codebook_loss_weight * codebook_loss

        # Backward pass
        total_loss.backward()

        should_log = log_interval and (
            (batch + 1) % log_interval == len(train_dataloader) - 1
        )
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

            if wandb_run:
                wandb_run.log(log_dict)

            logger.info(f"Batch: {batch + 1} / {len(train_dataloader)}")
            logger.info(log_dict)

    codebook = model.codebook
    codebook_statistics = codebook.get_statistics()
    codebook.reset_statistics()

    return codebook_statistics


def validate_epoch_ema_codebook(
    model: nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    num_classes: int,
    device: torch.device,
) -> dict[str, Any]:
    """Validate the model with an EMA codebook for a single epoch."""

    model.eval()

    # Initialize torchmetrics for validation
    top1_accuracy = Accuracy(task="multiclass", num_classes=num_classes, top_k=1).to(
        device
    )
    top5_accuracy = Accuracy(task="multiclass", num_classes=num_classes, top_k=5).to(
        device
    )

    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = (
                inputs.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )

            logits, _ = model(inputs)

            # Update metrics
            top1_accuracy.update(logits, labels)
            top5_accuracy.update(logits, labels)

    # Compute final accuracies
    top1_acc = top1_accuracy.compute() * 100
    top5_acc = top5_accuracy.compute() * 100

    codebook = model.codebook

    # Convert to Python scalars only for logging/return
    codebook_statistics = codebook.get_statistics()
    codebook_statistics["Top1 Accuracy"] = top1_acc.item()
    codebook_statistics["Top5 Accuracy"] = top5_acc.item()

    logger.info(
        f"Validation - Top1: {top1_acc.item():.2f}%, Top5: {top5_acc.item():.2f}%"
    )

    return codebook_statistics
