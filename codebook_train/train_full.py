# train.py
import datetime
import argparse
from pathlib import Path
import os
import torch.multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import models
from src.datasets.transforms import get_deit_transforms, get_default_image_transforms
from src.datasets.construct_dataset import get_dataset
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
import torch.distributed as dist

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
# New: use shared sampler helpers
from src.distributed_utils import create_samplers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ViTWithGAP(nn.Module):
    """
    Wrapper for torchvision VisionTransformer that replaces class-token
    classification with Global Average Pooling (GAP) over patch tokens.
    Mechanics:
      - torchvision's _process_input() returns patch embeddings WITHOUT the CLS token.
      - We prepend the learned CLS token -> [CLS, P1, P2, ..., Pn].
      - We run the encoder on that sequence.
      - We discard the CLS token and average only the patch tokens: GAP = mean(P1..Pn).
      - The pooled vector is passed through the original classification head.
    This is proper GAP over spatial (patch) tokens; the class token is excluded.
    """
    def __init__(self, vit: nn.Module):
        super().__init__()
        self.vit = vit

    def forward(self, x):
        # Get patch embeddings (no CLS yet)
        x = self.vit._process_input(x)  # shape: (B, N_patches, C)
        # Prepend CLS token to match positional embedding shape (1 + N_patches)
        n = x.shape[0]
        cls = self.vit.class_token.expand(n, -1, -1)  # (B, 1, C)
        x = torch.cat((cls, x), dim=1)                # (B, 1 + N_patches, C)
        # Encode sequence
        x = self.vit.encoder(x)
        # Proper GAP over patch tokens only (exclude CLS at index 0)
        gap = x[:, 1:].mean(dim=1)                    # (B, C)
        # Classify
        return self.vit.heads(gap)

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
    "imagenet1k": 1000,
}

WANDB_PROJECT="full-training"
WANDB_ENTITY="bubuss"

# ======================================================================================
# 2. Main Training Script
# ======================================================================================

def main(args):
    logger.info(f"Starting training script with args: {args}")
    utils.set_reproducibility(args.seed)
    distributed = args.distributed
    # Removed auto world_size derivation
    if distributed:
        device = _init_distributed(args)
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        device = DEVICE
        rank = 0
        world_size = 1
    is_rank0 = (rank == 0)
    if is_rank0:
        logger.info(f"Running in {'DDP' if distributed else 'single'} mode; world_size={world_size}")
    args.checkpoint_path.mkdir(exist_ok=True)
    checkpoint_tracker = utils.CheckpointTracker()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.wandb and is_rank0:
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
    # Distributed samplers (use shared helper for consistency)
    if distributed:
        train_sampler, val_sampler = create_samplers(
            train_dataset, val_dataset, rank=rank, world_size=world_size, seed=args.seed
        )
    else:
        train_sampler = None
        val_sampler = None
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    if is_rank0:
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
    elif args.model_name == "vit_b_16":
        base = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        hidden = base.heads.head.in_features
        base.heads.head = nn.Linear(hidden, num_classes)
        model = ViTWithGAP(base)
        logger.info("Using Vision Transformer B/16 with GAP over patch tokens (excluding CLS).")
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    # Ensure the trainable model is on DEVICE
    model = model.to(device)
    # Wrap with DDP
    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            broadcast_buffers=True,
        )
    if is_rank0:
        logger.info(f"Model: {model.__class__.__name__}")
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

    logger.info("\nStarting training..." if is_rank0 else "")
    ckpt_path = None
    for epoch in range(args.num_epochs):
        if distributed:
            train_loader.sampler.set_epoch(epoch)  # type: ignore
            # (val sampler has shuffle=False so no need to set_epoch)
        if is_rank0:
            logger.info(f"--- Epoch {epoch + 1}/{args.num_epochs} ---")
        train_epoch(
            model=model,
            train_dataloader=train_loader,
            transforms=mixup_fn,  # type: ignore
            optimizers=[optimizer],
            criterion=criterion,
            device=device,
            wandb_run=wandb_run,
            schedulers=[scheduler],
            epoch=epoch,
        )
        # Validation (all ranks if distributed for speed) - metrics aggregated via all_reduce
        if distributed:
            top1_acc, top5_acc = _distributed_validate(model, val_loader, device)
        else:
            top1_acc, top5_acc = validate_epoch(
                model=model,
                val_dataloader=val_loader,
                device=device
            )

        if is_rank0:
            logger.info(
                f"Validation | Top-1 Accuracy: {top1_acc:.2f}% | Top-5 Accuracy: {top5_acc:.2f}%"
            )
            if checkpoint_tracker.is_best(top1_acc):
                logger.info(f"Saving the best checkpoint at epoch: {epoch + 1} with accuracy: {top1_acc:.2f}%")
                ckpt_path = (
                    args.checkpoint_path
                    / f"{args.dataset}_{args.model_name}_full_{timestamp}.pth"
                )
                state_dict = model.module.state_dict() if distributed else model.state_dict()
                torch.save(state_dict, ckpt_path)

        # Step scheduler on all ranks
        scheduler.step(epoch + 1)
        if is_rank0 and wandb_run:
            current_lr = optimizer.param_groups[0]["lr"]
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "val_top1": top1_acc,
                    "val_top5": top5_acc,
                    "lr": current_lr,
                }
            )
    if is_rank0 and wandb_run and ckpt_path:
        wandb_run.save(ckpt_path, policy="now")
        logger.info("Training finished.")
        logger.info(
            f"Best validation Top-1 accuracy: {checkpoint_tracker.best_val_accuracy:.2f}%"
        )
        abs_path = ckpt_path.resolve()
        logger.info(f"Saved at: {abs_path}")
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

# --- New helpers for DDP setup and distributed validation ---

def _init_distributed(args) -> torch.device:
    """Initialize torch.distributed from argparse args and select device."""
    # Set env (env:// is recommended)
    os.environ.setdefault("MASTER_ADDR", str(args.master_addr))
    os.environ.setdefault("MASTER_PORT", str(args.master_port))
    os.environ.setdefault("RANK", str(args.rank))
    os.environ.setdefault("WORLD_SIZE", str(args.world_size))
    os.environ.setdefault("LOCAL_RANK", str(args.local_rank))

    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"

    if use_cuda:
        device = torch.device("cuda", args.local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=int(os.environ["WORLD_SIZE"]),
            rank=int(os.environ["RANK"]),
        )
    dist.barrier()
    return device


@torch.no_grad()
def _distributed_validate(model: nn.Module, val_loader: DataLoader, device: torch.device):
    """Validate across distributed shards and return globally reduced Top-1/Top-5."""
    was_training = model.training
    model.eval()

    total = torch.tensor(0, device=device, dtype=torch.long)
    correct1 = torch.tensor(0, device=device, dtype=torch.long)
    correct5 = torch.tensor(0, device=device, dtype=torch.long)

    for images, targets in val_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        # Top-1
        pred1 = outputs.argmax(dim=1)
        correct1 += (pred1 == targets).sum()
        # Top-5 (or top-k if classes < 5)
        k = min(5, outputs.size(1))
        _, topk = outputs.topk(k, dim=1, largest=True, sorted=True)
        correct5 += (topk.eq(targets.view(-1, 1))).any(dim=1).sum()

        total += targets.size(0)

    # Global reduction
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct1, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct5, op=dist.ReduceOp.SUM)

    top1 = (correct1.float() / total.clamp_min(1).float() * 100.0).item()
    top5 = (correct5.float() / total.clamp_min(1).float() * 100.0).item()

    if was_training:
        model.train()
    return top1, top5

def _spawn_worker(local_rank: int, args):
    """Spawned process entrypoint."""
    args.local_rank = local_rank
    args.rank = local_rank  # single-node assumption
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["RANK"] = str(args.rank)
    os.environ["LOCAL_RANK"] = str(args.local_rank)
    os.environ["MASTER_ADDR"] = str(args.master_addr)
    os.environ["MASTER_PORT"] = str(args.master_port)
    main(args)

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
        choices=["stanford_cars", "flowers102", "cub200", "stanford_dogs", "imagenet1k"],
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
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable Distributed Data Parallel.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local GPU index for this process.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Global rank of this process.",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Total number of processes to participate in the job.",
    )
    parser.add_argument(
        "--master_addr",
        type=str,
        default="127.0.0.1",
        help="Master node address for TCP initialization.",
    )
    parser.add_argument(
        "--master_port",
        type=str,
        default="29500",
        help="Master node port for TCP initialization.",
    )

    args = parser.parse_args()
    if args.distributed:
        mp.set_start_method("spawn", force=True)
        mp.spawn(_spawn_worker, nprocs=args.world_size, args=(args,))
    else:
        main(args)
