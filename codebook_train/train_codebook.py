import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from codebook import Codebook
from utils import train_epoch, validate_epoch, set_reproducibility
from imagenet import get_imagenet

import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None
    logger.info("wandb is not available, skipping wandb.init")


def codebook_training(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    wandb_run=None,
):
    for epoch in range(epochs):
        logger.info(f"Epoch: {epoch}")
        train_epoch(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            wandb_run=wandb_run,
        )

        top1_acc, top5_acc = validate_epoch(
            model=model, val_dataloader=val_dataloader, device=device
        )

        logger.info(f"Validation top1-accuracy {top1_acc}, top5-accuracy {top5_acc}")
        if wandb:
            wandb.log(
                {
                    "Validation Top1 Accuracy": top1_acc,
                    "Validation Top5 Accuracy": top5_acc,
                }
            )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--epochs", type=int, required=True)
    argparser.add_argument("--batch_size", type=int, required=True)
    argparser.add_argument("--imagenet_path", type=str, required=True)
    argparser.add_argument("--codebook_size", type=int, required=True)
    argparser.add_argument("--lr", type=float, default=0.001)
    argparser.add_argument("--weight_decay", type=float, default=0.00002)
    argparser.add_argument("--num_workers", type=int, default=8)
    argparser.add_argument("--embedding_dim", type=int, default=768)
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument("--wandb_project", type=str, default="codebook-training")

    args = argparser.parse_args()
    logger.info(f"Config: {args}")
    set_reproducibility(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    convnext_tiny = models.convnext_tiny(
        weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    )
    convnext_tiny = convnext_tiny.to(device)
    codebook = Codebook(args.codebook_size, args.embedding_dim).to(device)
    # inject the codebook into the model after features
    convnext_tiny.features.add_module("codebook", codebook)

    # Set requires_grad to False for all parameters
    for param in convnext_tiny.parameters():
        param.requires_grad = False

    # Set requires_grad to True for the codebook parameters
    for param in codebook.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(
        codebook.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    logger.info(f"Model: {convnext_tiny}")

    train_dataloader, val_dataloader = get_imagenet(
        args.imagenet_path, args.batch_size, args.num_workers
    )

    if wandb:
        wandb_run = wandb.init(project=args.wandb_project, config=args)
    else:
        wandb_run = None

    codebook_training(
        model=convnext_tiny,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        wandb_run=wandb_run,
    )
