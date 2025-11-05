# train.py
import datetime
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from src.datasets.transforms import get_deit_transforms, get_default_image_transforms
from src.datasets.construct_dataset import get_dataset
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy

try:
    import wandb
except ImportError:
    wandb = None


# Import utility functions from utils.py
import src.utils as utils

from src.training import train_epoch, validate_epoch
# Import data loading functions from the new 'datasets' package
from src.datasets.cub import get_cub200
from src.datasets.flowers102 import get_flowers102
from src.datasets.stanford_cars import get_stanford_cars
from src.datasets.stanford_dogs import get_stanford_dogs
from src.models.deit import deit_small_patch16_224
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

WANDB_PROJECT="full-training"
WANDB_ENTITY="bubuss"

# ======================================================================================
# 2. Main Training Script
# ======================================================================================


def main(args):
    logger.info(f"Starting training script with args: {args}")
    # --- Setup for Reproducibility and Checkpoints ---
    utils.set_reproducibility(args.seed)
    args.checkpoint_path.mkdir(exist_ok=True)
    checkpoint_tracker = utils.CheckpointTracker()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Install with: pip install wandb")
        run_name = f"full_{args.dataset}_{args.model_name}_{timestamp}"
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            config=vars(args),
        )
    else:
        wandb_run = None


    logger.info(f"Loading '{args.dataset}' dataset...")
    num_classes = NUM_CLASSES[args.dataset]
    logger.info(f"Number of classes: {num_classes}")
    
    if args.transforms == "deit":
        train_transform, val_transform = get_deit_transforms(is_precropped=(args.dataset == "cub200"))
        logger.info("Using Deit image transforms.")
    else:
        train_transform, val_transform = get_default_image_transforms(
            autoaugment=args.autoaugment,
            resize_value=224 if args.dataset == "cub200" else 256,
            crop_value=None if args.dataset == "cub200" else 224,
            random_erase=0.1,
            horizontal_flip=0.5,
            is_precropped=args.dataset == "cub200",
        )
        logger.info("Using default image transforms.")
            
    train_dataset, val_dataset = get_dataset(
        name=args.dataset, 
        path=args.data_path, 
        train_transform=train_transform, 
        val_transform=val_transform
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
    logger.info(f"Loading {args.model_name} and preparing for fine-tuning...")
    # --- Select and Load Data ---
    if args.model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif args.model_name == "convnext_large":
        model = models.convnext_large(
            weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1
        )
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif args.model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model_name == "deit_small_patch16_224":
        model = deit_small_patch16_224(pretrained=True)
        global_pool = "avg"
        model.reset_classifier(num_classes=num_classes, global_pool=global_pool)
        logger.info(f"Global pool set to {global_pool} for Deit model.")
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    # Ensure the trainable model is on DEVICE
    model = model.to(DEVICE)
    logger.info(f"Model: {model}")

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    logger.info(f"Optimizer: {optimizer}")
    scheduler_args = utils.SchedulerArgs(
        epochs=args.num_epochs,
        warmup_epochs=args.warmup_epochs,
        lr=args.learning_rate
    )
    scheduler = utils.create_schedulers(
        optimizers=[optimizer],
        scheduler_args=scheduler_args
    )[0]

    # --- Define Loss, Optimizer, and Transforms for training loop ---
    # Use softer augmentation for CUB and match loss to mixup soft targets
    mixup_fn = Mixup(
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        cutmix_minmax=None,
        prob=args.mixup_prob,
        switch_prob=args.switch_prob,
        mode='batch',
        label_smoothing=args.label_smoothing,
        num_classes=num_classes
    )
    criterion = SoftTargetCrossEntropy()

    logger.info("\nStarting training...")
    ckpt_path = None
    # --- Training & Validation Loop using utils ---
    for epoch in range(args.num_epochs):
        logger.info(f"--- Epoch {epoch + 1}/{args.num_epochs} ---")

        train_epoch(
            model=model,
            train_dataloader=train_loader,
            transforms=mixup_fn, # type: ignore
            optimizers=[optimizer],
            criterion=criterion,
            device=DEVICE,
            wandb_run=wandb_run,
            schedulers=[scheduler],
            epoch=epoch,
        )

        # Validate using raw model weights
        top1_acc, top5_acc = validate_epoch(
            model=model, val_dataloader=val_loader, device=DEVICE
        )

        logger.info(
            f"Validation | Top-1 Accuracy: {top1_acc:.2f}% | Top-5 Accuracy: {top5_acc:.2f}%"
        )

        # --- Scheduler Step (per-epoch) ---
        scheduler.step(epoch + 1)
        current_lr = optimizer.param_groups[0]["lr"]
        
        if wandb_run:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "val_top1": top1_acc,
                    "val_top5": top5_acc,
                    "lr": current_lr,
                }
            )

        if checkpoint_tracker.is_best(top1_acc):
            logger.info(f"Saving the best checkpoint at epoch: {epoch + 1} with accuracy: {top1_acc:.2f}%")
            ckpt_path = (
                args.checkpoint_path
                / f"{args.dataset}_{args.model_name}_full_{timestamp}.pth"
            )
            logger.info(f"Timestamp of the checkpoint {timestamp}, path {ckpt_path}")
            torch.save(
                model.state_dict(),
                ckpt_path,
            )
            
    if wandb_run and ckpt_path:
        wandb_run.save(
            ckpt_path,
            policy="now",
        )

    logger.info("Training finished.")
    logger.info(
        f"Best validation Top-1 accuracy: {checkpoint_tracker.best_val_accuracy:.2f}%"
    )
    logger.info(f"Saved at {args.dataset}_{args.model_name}_full_{timestamp}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model using utility functions."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model to train.",
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
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for the optimizer.",
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
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--autoaugment",
        action="store_true",
        help="Use AutoAugment instead of TrivialAugmentWide.",
    )
    parser.add_argument(
        "--transforms",
        type=str,
        choices=["deit", "default"],
        default="default",
        help="Which transforms pipeline to use.",
    )
    parser.add_argument("--mixup_alpha", type=float, default=0.8, help="Mixup alpha.")
    parser.add_argument("--cutmix_alpha", type=float, default=1.0, help="CutMix alpha.")
    parser.add_argument("--mixup_prob", type=float, default=1.0, help="Probability of applying mixup/cutmix.")
    parser.add_argument("--switch_prob", type=float, default=0.5, help="Switch probability between mixup and cutmix.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing for soft targets.")

    args = parser.parse_args()
    main(args)
