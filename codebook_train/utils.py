from dataclasses import asdict
import logging
import torch
import torch.nn as nn
import random
import numpy as np

from config.optimizers import BaseOptimizerConfig

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

            outputs = model(inputs)
            _, pred = outputs.topk(5, 1, True, True)

            total += labels.size(0)
            top1_correct += (pred[:, :1] == labels.view(-1, 1)).sum().item()
            top5_correct += (pred == labels.view(-1, 1)).sum().item()

    top1_acc = (top1_correct / total) * 100
    top5_acc = (top5_correct / total) * 100
    return top1_acc, top5_acc


def train_epoch(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    optimizers: list[torch.optim.Optimizer],
    criterion: nn.Module,
    device: torch.device,
    wandb_run=None,
) -> None:
    model.train()

    for i, (images, labels) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device)

        for optimizer in optimizers:
            optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)
        loss.backward()

        for optimizer in optimizers:
            optimizer.step()

        accuracy = (output.argmax(1) == labels).float().mean()

        if wandb_run:
            wandb_run.log(
                {"Train Loss": loss.item(), "Train Accuracy": accuracy.item()}
            )

        if i % (len(train_dataloader) // 10) == 0:
            logger.info(
                f"Iteration: {i} / {len(train_dataloader)}, Loss: {loss.item()}, Accuracy: {accuracy.item()}"
            )


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

    checkpoint = {"model": model.state_dict()}
    torch.save(checkpoint, path)


def get_param_groups(
    base_optimizer_cfg: BaseOptimizerConfig,
    codebook_optimizer_cfg: BaseOptimizerConfig,
    base_parameters: list[torch.nn.Parameter],
    codebook_parameters: list[torch.nn.Parameter],
) -> list[dict[str, any]]:
    base_optimizer_dict = asdict(base_optimizer_cfg)
    codebook_optimizer_dict = asdict(codebook_optimizer_cfg)
    del base_optimizer_dict["_target_"]
    del codebook_optimizer_dict["_target_"]

    base_optimizer_dict["params"] = base_parameters
    codebook_optimizer_dict["params"] = codebook_parameters

    param_groups = [base_optimizer_dict, codebook_optimizer_dict]
    return param_groups
