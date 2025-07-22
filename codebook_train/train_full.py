# train.py
import datetime
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import torchvision.transforms.v2 as transforms_v2


# Import utility functions from utils.py
import src.utils as utils

# Import data loading functions from the new 'datasets' package
from src.datasets.cub import get_cub200
from src.datasets.flowers102 import get_flowers102
from src.datasets.stanford_cars import get_stanford_cars
from src.datasets.stanford_dogs import get_stanford_dogs
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================================================================om ==============
# 1. Configuration
# ======================================================================================

# --- Paths & Model Parameters ---
NUM_WORKERS = 8

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

NUM_CLASSES = {
    "stanford_cars": 196,
    "flowers102": 102,
    "cub200": 200,
    "stanford_dogs": 120,
}


# ======================================================================================
# 2. Main Training Script
# ======================================================================================


def main(args):
    logger.info(f"Starting training script with args: {args}")
    # --- Setup for Reproducibility and Checkpoints ---
    utils.set_reproducibility(args.seed)
    args.checkpoint_path.mkdir(exist_ok=True)
    checkpoint_tracker = utils.CheckpointTracker()

    # --- Select and Load Data ---
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info(f"Loading '{args.dataset}' dataset...")
    if args.dataset == "stanford_cars":
        loader_fn = get_stanford_cars
    elif args.dataset == "flowers102":
        loader_fn = get_flowers102
    elif args.dataset == "cub200":
        loader_fn = get_cub200
    elif args.dataset == "stanford_dogs":
        loader_fn = get_stanford_dogs
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    num_classes = NUM_CLASSES[args.dataset]
    logger.info(f"Number of classes: {num_classes}")

    train_dataset, val_dataset = loader_fn(
        path=args.data_path,
        resize_value=256,
        crop_value=224,
        random_erase=0.1,
        horizontal_flip=0.5,
    )
    logger.info(f"Dataset loaded. Found {num_classes} classes.")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    logger.info(f"Train loader size: {len(train_loader)} batches")

    # --- Initialize Model ---
    logger.info("Loading ConvNeXt Tiny and preparing for fine-tuning...")
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    model = model.to(DEVICE)
    logger.info(f"Model: {model}")

    #for param in model.features.parameters():
        #param.requires_grad = False
        
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = utils.create_schedulers(
        optimizers=[optimizer],
        epoch_iters=len(train_loader),
        warmup_epochs=args.warmup_epochs,
        epochs=args.num_epochs,
    )[0]

    # --- Define Loss, Optimizer, and Transforms for training loop ---
    criterion = nn.CrossEntropyLoss()
    cutmix = transforms_v2.CutMix(num_classes=num_classes)
    mixup = transforms_v2.MixUp(num_classes=num_classes)
    cutmix_or_mixup = transforms_v2.RandomChoice([cutmix, mixup])

    logger.info("\nStarting training...")
    # --- Training & Validation Loop using utils ---
    for epoch in range(args.num_epochs):
        logger.info(f"--- Epoch {epoch+1}/{args.num_epochs} ---")

        utils.train_epoch(
            model=model,
            train_dataloader=train_loader,
            transforms=cutmix_or_mixup,
            optimizers=[optimizer],
            criterion=criterion,
            device=DEVICE,
            wandb_run=None,
        )

        top1_acc, top5_acc = utils.validate_epoch(
            model=model, val_dataloader=val_loader, device=DEVICE
        )

        logger.info(
            f"Validation | Top-1 Accuracy: {top1_acc:.2f}% | Top-5 Accuracy: {top5_acc:.2f}%"
        )
        
        # --- Scheduler Step ---
        scheduler.step()

        if checkpoint_tracker.is_best(top1_acc):
            
            torch.save(
                model.state_dict(),
                args.checkpoint_path / f"{args.dataset}_convnext_full_{timestamp}.pth",
            )

    logger.info("Training finished.")
    logger.info(
        f"Best validation Top-1 accuracy: {checkpoint_tracker.best_val_accuracy:.2f}%"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ConvNeXt using utility functions."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["stanford_cars", "flowers102", "cub200", "stanford_dogs"],
        help="The dataset to train on.",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training.",
    )
    parser.add_argument(
        "-e",
        "--num_epochs",
        type=int,
        help="Number of epochs to train the model.",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Number of warmup epochs for the learning rate scheduler.",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("./data"),
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("./checkpoints"),
        help="Path to save model checkpoints.",
    )

    args = parser.parse_args()
    main(args)
