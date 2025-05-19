from typing import Any
import torch
import torch.nn as nn
import faiss
from torch.utils.data import DataLoader
from src.codebook_wrappers import CNNCodebookWrapper
from src.utils import calculate_accuracy
import numpy as np
import logging

logger = logging.getLogger(__name__)


def evaluate_linear_probe(
    model: nn.Module,
    train_dl: DataLoader,
    train_transforms: nn.Module,
    val_dl: DataLoader,
    linear_probe: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epochs: int,
    criterion: nn.Module,
    apply_adaptive_pooling: bool = True,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    wandb_run=None,
) -> float:
    """
    Evaluates the linear probe model on the validation set.

    Args:
        model (nn.Module): The feature encoder model.
        train_dl (DataLoader): DataLoader for the training set.
        train_transforms (nn.Module): Transforms to apply to training data.
        val_dl (DataLoader): DataLoader for the validation set.
        linear_probe (nn.Module): The linear probe model.
        optimizer (torch.optim.Optimizer): Optimizer for the linear probe.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler for the optimizer.
        epochs (int): Number of epochs to train the linear probe.
        criterion (nn.Module): Loss function for training.
        apply_adaptive_pooling (bool): Whether to apply adaptive pooling to features.
        device (torch.device): Device to run the model on ('cuda' or 'cpu').
        wandb_run: Wandb object for logging (optional).

    Returns:
        float: Validation accuracy.
    """

    model, linear_probe = model.to(device), linear_probe.to(device)

    for epoch in range(epochs):
        logger.info(f"Linear Probe Training Epoch {epoch}/{epochs}")
        model.train()
        linear_probe.train()

        for examples, labels in train_dl:
            examples, labels = examples.to(device), labels.to(device)
            examples, labels = train_transforms(examples, labels)

            optimizer.zero_grad()
            with torch.no_grad():
                features = model(examples)
                if isinstance(features, tuple):
                    features = features[0]

                if apply_adaptive_pooling:
                    features = features.mean(dim=[2, 3])  # Global average pooling

            logits = linear_probe(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

        # Evaluate on validation set
        model.eval()
        linear_probe.eval()
        mean_accuracy = 0.0
        for examples, labels in val_dl:
            examples, labels = examples.to(device), labels.to(device)

            with torch.no_grad():
                features = model(examples)
                if isinstance(features, tuple):
                    features = features[0]

                if apply_adaptive_pooling:
                    features = features.mean(dim=[2, 3])  # Global average pooling

                logits = linear_probe(features)
                accuracy = calculate_accuracy(logits, labels)
                mean_accuracy += accuracy

        mean_accuracy /= len(val_dl)
        logger.info(
            f"Epoch {epoch}/{epochs} Linear Probe Validation Accuracy: {mean_accuracy:.4f}%"
        )
        if wandb_run:
            wandb_run.log(
                {
                    "Probe Epoch": epoch,
                    "Probe Validation Accuracy": mean_accuracy,
                }
            )

    return mean_accuracy


@torch.no_grad()
def extract_features(
    model_encoder: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts features and labels from a dataloader using the given encoder.

    Args:
        model_encoder (nn.Module): The feature encoder model.
        dataloader (DataLoader): DataLoader for the dataset.
        device (str): Device to run the model on ('cuda' or 'cpu').

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - features_tensor: Tensor of extracted features.
            - labels_tensor: Tensor of corresponding labels.
    """

    is_train = model_encoder.training
    if is_train:
        model_encoder.eval()

    features_list = []
    labels_list = []

    for images, labels in dataloader:
        images = images.to(device)
        out = model_encoder(images)
        if isinstance(out, tuple):
            features: torch.Tensor = out[0]
        else:
            features: torch.Tensor = out

        features = features.mean(dim=[2, 3])  # Global average pooling
        features = nn.functional.normalize(features, dim=1)

        features_list.append(features.cpu())
        labels_list.append(labels)

    features_tensor = torch.cat(features_list, dim=0)
    labels_tensor = torch.cat(labels_list, dim=0)

    if is_train:
        model_encoder.train()

    return features_tensor, labels_tensor


def evaluate_knn(
    model_encoder: nn.Module,
    train_dataloader_knn: torch.utils.data.DataLoader,
    val_dataloader_knn: torch.utils.data.DataLoader,
    k: int = 10,
    device: str = "cuda",
):
    logger.info("Extracting training features...")
    train_features, train_labels = extract_features(
        model_encoder, train_dataloader_knn, device
    )
    train_features_np = train_features.cpu().numpy()
    train_labels_np = train_labels.cpu().numpy()

    logger.info("Extracting validation features...")
    val_features, val_labels = extract_features(
        model_encoder, val_dataloader_knn, device
    )
    val_features_np = val_features.cpu().numpy()
    val_labels_np = val_labels.cpu().numpy()

    d = train_features_np.shape[1]  # Dimension of features

    logger.info("Building Faiss index...")
    # Simple flat L2 index (exact search, good for moderate size or GPU)
    index = faiss.IndexFlatL2(d)

    gpu_resource = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(gpu_resource, 0, index)

    index.add(train_features_np.astype(np.float32))  # Add gallery features to index
    logger.info(f"Faiss index built with {index.ntotal} vectors.")

    logger.info(f"Searching KNN (k={k}) with Faiss...")
    # D: distances, I: indices of neighbors in the gallery
    _, indices = index.search(val_features_np.astype(np.float32), k)
    logger.info(indices)
    predictions = []
    for neighbors_indices in indices:
        neighbor_actual_labels = train_labels_np[neighbors_indices]
        # Simple majority vote
        pred_label = np.bincount(neighbor_actual_labels).argmax()
        predictions.append(pred_label)

    predictions_np = np.array(predictions)
    correct = (predictions_np == val_labels_np).sum()
    accuracy = correct / len(val_labels_np)

    return accuracy * 100


def train_epoch_ssl(
    model: CNNCodebookWrapper,
    train_dataloader: torch.utils.data.DataLoader,
    transforms: torch.nn.Module,
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    wandb_run=None,
) -> dict[str, Any]:
    """Train a single epoch of the model with cosine codebook

    Args:
        model (CNNCodebookWrapper): Wrapper around ConvNext model with cosine codebook
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data
        transforms (torch.nn.Module): Transformations to apply to the input data
        optimizers (list[torch.optim.Optimizer]): List of optimizers to use
        schedulers (list[torch.optim.lr_scheduler._LRScheduler]): List of schedulers to use
        device (torch.device): Device to use for training
        wandb_run (_type_, optional): Wandb object for logging. Defaults to None.

    Returns:
        dict[str, float]: Statistics of the codebook after training
    """

    model.train()

    for batch, (images, labels) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        transformed_images, _ = transforms(images, labels)

        for optimizer in optimizers:
            optimizer.zero_grad()

        _, codebook_loss = model(transformed_images)

        # Backward pass
        codebook_loss.backward()

        for optimizer in optimizers:
            optimizer.step()

        # Step schedulers after each epoch
        for scheduler in schedulers:
            scheduler.step()

        if (batch + 1) % (len(train_dataloader) // 10) == 0:
            log_dict = {
                "Codebook Loss": codebook_loss.item(),
            }

            if wandb_run:
                wandb_run.log(log_dict)

            logger.info(f"Batch: {batch + 1} / {len(train_dataloader)}")
            logger.info(log_dict)

    codebook_statistics = model.codebook.get_statistics()
    model.codebook.reset_statistics()

    return codebook_statistics


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    epoch_iters: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create schedulers for the optimizers

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to create schedulers for
        epochs (int): Number of epochs
        epoch_iters (int): Number of iterations per epoch
        warmup_epochs (int): Number of warmup epochs

    Returns:
        torch.optim.lr_scheduler._LRScheduler: Scheduler for the optimizer
    """

    total_iterations = epoch_iters * epochs
    logger.info(f"Total iterations: {total_iterations}")

    linear_scheduler_iters = warmup_epochs * epoch_iters
    linear_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=linear_scheduler_iters,
    )

    cosine_cycles = 3
    remaining_iterations = total_iterations - linear_scheduler_iters
    initial_cycle_length = remaining_iterations // (2**cosine_cycles - 1)

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=initial_cycle_length,
        T_mult=2,
        eta_min=0.000001,
    )
    sequential_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[linear_scheduler, cosine_scheduler],
        milestones=[linear_scheduler_iters],
    )

    logger.info(f"Warmup iterations: {linear_scheduler_iters}")
    logger.info(
        f"Cosine iterations: {[initial_cycle_length * 2**i for i in range(1, cosine_cycles + 1)]}"
    )

    return sequential_scheduler
