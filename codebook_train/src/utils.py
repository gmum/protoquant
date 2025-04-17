import logging
import torch
import torch.nn as nn
import random
import numpy as np
from src.codebook import ConvNextCosineWrapper
from src.config.main_config import MainConfig
import hydra

logging.basicConfig(level=logging.INFO)
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
    model: ConvNextCosineWrapper,
    train_dataloader: torch.utils.data.DataLoader,
    transforms: torch.nn.Module,
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[torch.optim.lr_scheduler._LRScheduler],
    criterion: nn.Module,
    task_loss_weight: float,
    codebook_loss_weight: float,
    device: torch.device,
    restart_threshold: int,
    wandb_run=None,
) -> dict[str, float]:
    """Train a single epoch of the model with cosine codebook

    Args:
        model (ConvNextCosineWrapper): Wrapper around ConvNext model with cosine codebook
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data
        transforms (torch.nn.Module): Transformations to apply to the input data
        optimizers (list[torch.optim.Optimizer]): List of optimizers to use
        schedulers (list[torch.optim.lr_scheduler._LRScheduler]): List of schedulers to use
        criterion (nn.Module): Loss function to use
        task_loss_weight (float): Weight for the task loss
        codebook_loss_weight (float): Weight for the codebook loss
        device (torch.device): Device to use for training
        restart_threshold (int): Threshold for restarting the codebook.
        wandb_run (_type_, optional): Wandb object for logging. Defaults to None.

    Returns:
        dict[str, float]: Statistics of the codebook after training
    """

    model.train()

    for batch, (images, labels) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        transformed_images, transformed_labels = transforms(images, labels)

        for optimizer in optimizers:
            optimizer.zero_grad()

        logits, codebook_loss = model(transformed_images)
        task_loss = criterion(logits, transformed_labels)

        total_loss = task_loss_weight * task_loss + codebook_loss_weight * codebook_loss

        # Backward pass
        total_loss.backward()

        for optimizer in optimizers:
            optimizer.step()

        # Step schedulers after each epoch
        for scheduler in schedulers:
            scheduler.step()

        if (batch + 1) % (len(train_dataloader) // 10) == 0:
            accuracy = (logits.argmax(1) == labels).float().mean()
            log_dict = {
                "Train Loss": total_loss.item(),
                "Task Loss": task_loss.item(),
                "Codebook Loss": codebook_loss.item(),
                "Train Accuracy": accuracy.item(),
            }

            if wandb_run:
                wandb_run.log(log_dict)

            logger.info(f"Batch: {batch + 1} / {len(train_dataloader)}")
            logger.info(log_dict)

    if restart_threshold >= 0:
        model.codebook.restart_codes(restart_threshold)

    codebook_statistics = model.codebook.get_statistics()
    model.codebook.reset_statistics()

    return codebook_statistics


def validate_epoch_cosine_codebook(
    model: nn.Module, val_dataloader: torch.utils.data.DataLoader, device: torch.device
) -> tuple[float, float]:
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
    return top1_acc, top5_acc


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


def save_checkpoint(path: str, model: nn.Module) -> None:
    """Save the model state to a checkpoint file

    Args:
        path (str): The path to save the checkpoint file
        model (nn.Module): The model to save
    """

    torch.save(model.state_dict(), path)


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
