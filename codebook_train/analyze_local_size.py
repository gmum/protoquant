# analyze_local_size.py

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

# --- Add project's source to the Python path for imports ---
# This ensures that modules from the 'src' directory can be found.
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
# -----------------------------------------------------------

# --- Project-specific Imports ---
# Import necessary classes and functions from the project structure.
from src.config.pipnet_config import BaseDatasetConfig, PipNetConfig
from src.datasets.construct_dataset import get_dataset
from src.datasets.transforms import get_default_image_transforms
from src.models.pipnet import QuantizedPIPNetHead
from src.pipnet_utils import PIPNetWrapper, TrainingWrapper, build_pipnet_model

# --- Basic Logger Setup ---
# Configure a logger for clear and timed console output.
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def print_local_size_report(
    pipnet_checkpoint_path: Path,
    model_name: str,
    num_classes: int,
    head,
    threshold: float,
    local_sizes: torch.Tensor,
) -> None:
    print("\n" + "=" * 60)
    print("      Prototype Local Size Analysis Report")
    print("=" * 60)
    print(f"Model Checkpoint:       {pipnet_checkpoint_path.name}")
    print(f"Backbone:               {model_name}")
    print(f"Number of Classes:      {num_classes}")
    print(f"Total Prototypes (P):   {head.P}")
    print(f"Significance Threshold: {threshold}")
    print("-" * 60)

    logger.info("Analyzing classifier weight distribution...")
    with torch.no_grad():
        positive_weights = torch.relu(head.classifier.weight)

        print("\n" + "-" * 60)
        print("      Classifier Weight Distribution Analysis")
        print("-" * 60)
        print(f"  Shape of weights: {positive_weights.shape}")
        print(f"  Mean of all positive weights:   {positive_weights.mean().item():.4f}")
        print(f"  Std Dev of all positive weights: {positive_weights.std().item():.4f}")
        print(f"  Min of all positive weights:    {positive_weights.min().item():.4f}")
        print(f"  Max of all positive weights:    {positive_weights.max().item():.4f}")
        print(f"  Median of all positive weights: {torch.median(positive_weights).item():.4f}")
        print("-" * 60)

    for i, size in enumerate(local_sizes):
        print(f"  Class {i:3d}: {int(size.item()):4d} significant prototypes")

    avg_size = local_sizes.float().mean().item()
    min_size = int(local_sizes.min().item())
    max_size = int(local_sizes.max().item())

    with torch.no_grad():
        positive_weights = torch.relu(head.classifier.weight.detach())
        is_significant_mask = positive_weights > threshold
        unique_protos_used = int(is_significant_mask.any(dim=0).sum().item())

    print("-" * 60)
    print("Summary Statistics:")
    print(f"  Average local size per class: {avg_size:.2f}")
    print(f"  Min | Max local size:         {min_size} | {max_size}")
    print(f"  Total unique prototypes used: {unique_protos_used} / {head.P} ({ (unique_protos_used / head.P) * 100:.2f}%)")
    print("=" * 60 + "\n")

@torch.no_grad()
def validate_top_k(
    model: nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """
    Performs a validation run on a given dataloader and returns top-1 accuracy.
    This is a simplified, non-distributed validation loop for analysis.

    Args:
        model (nn.Module): The model to evaluate.
        val_dataloader (torch.utils.data.DataLoader): The dataloader for validation data.
        device (torch.device): The device (e.g., 'cuda:0') to run inference on.

    Returns:
        float: The top-1 accuracy percentage.
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    # Use tqdm for a progress bar during validation
    pbar = tqdm(val_dataloader, desc="Validating", leave=False, ncols=100)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # The TrainingWrapper ensures model(images) returns logits directly
        logits = model(images)

        # Get the index of the max log-probability (the predicted class)
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def evaluate_with_top_k_prototypes(
    model: PIPNetWrapper,
    head: QuantizedPIPNetHead,
    val_loader: torch.utils.data.DataLoader,
    top_k_list: list[int],
    device: torch.device,
    threshold: float,
) -> tuple[dict[str | int, float], dict[str | int, int], dict[str | int, float]]:
    """
    Evaluates the model's accuracy, first with all prototypes (full model) and then
    using only the top-k most influential prototypes for each class.

    Returns:
        A tuple containing:
        - dict[str | int, float]: A dictionary mapping k (or 'Full') to validation accuracy.
        - dict[str | int, int]: A dictionary mapping k (or 'Full') to unique prototypes used.
        - dict[str | int, float]: A dictionary mapping k (or 'Full') to avg local size per class.
    """
    logger.info(f"Starting top-k prototype evaluation for k in {top_k_list}...")

    eval_model = TrainingWrapper(model).to(device)
    original_weights = head.classifier.weight.data.clone()
    positive_weights = torch.relu(original_weights)

    results: dict[str | int, float] = {}
    unique_protos_counts: dict[str | int, int] = {}
    avg_local_sizes: dict[str | int, float] = {}

    # --- Evaluate the full model first as a baseline ---
    logger.info("--- Evaluating Full Model (Baseline) ---")
    full_model_accuracy = validate_top_k(eval_model, val_loader, device)
    full_model_protos_used = (positive_weights > 0).any(dim=0).sum().item()
    # Avg local size per class at current threshold
    full_local_sizes = head.calculate_local_size(threshold=threshold).float()
    avg_local_sizes['Full'] = full_local_sizes.mean().item()

    results['Full'] = full_model_accuracy
    unique_protos_counts['Full'] = full_model_protos_used
    
    logger.info(f"Full Model Validation Accuracy: {full_model_accuracy:.2f}%")
    logger.info(f"Unique Prototypes Used in Full Model: {full_model_protos_used} / {head.P}")
    logger.info(f"Avg local size/class (Full) @ thr={threshold}: {avg_local_sizes['Full']:.2f}")

    for k in top_k_list:
        if k > head.P:
            logger.warning(
                f"k={k} is larger than the total number of prototypes ({head.P}). Skipping."
            )
            continue

        logger.info(f"--- Evaluating with k = {k} ---")
        _, top_k_indices = torch.topk(positive_weights, k=k, dim=1)
        mask = torch.zeros_like(original_weights, device=device)
        mask.scatter_(1, top_k_indices, 1.0)
        masked_weights = original_weights * mask
        head.classifier.weight.data = masked_weights

        unique_protos_used = mask.any(dim=0).sum().item()
        unique_protos_counts[k] = unique_protos_used
        # Avg local size per class at threshold after masking
        k_local_sizes = head.calculate_local_size(threshold=threshold).float()
        avg_local_sizes[k] = k_local_sizes.mean().item()
        logger.info(
            f"Number of unique prototypes used for k={k}: {unique_protos_used} / {head.P}"
        )
        logger.info(
            f"Avg local size/class (k={k}) @ thr={threshold}: {avg_local_sizes[k]:.2f}"
        )

        accuracy = validate_top_k(eval_model, val_loader, device)
        results[k] = accuracy
        logger.info(f"Validation accuracy for k={k}: {accuracy:.2f}%")

    head.classifier.weight.data = original_weights
    logger.info("Restored original model weights.")

    return results, unique_protos_counts, avg_local_sizes


def plot_results(
    results: dict[str | int, float],
    unique_protos: dict[str | int, int],
    total_protos: int,
    output_dir: Path,
):
    """
    Plots the top-k evaluation results and saves the plot to a file.
    Includes the full model's performance as a baseline.
    """
    # Separate the 'Full' model results from the top-k results
    full_model_acc = results.pop('Full', None)
    unique_protos.pop('Full', None)

    k_values = sorted(results.keys())
    accuracies = [results[k] for k in k_values]
    unique_counts = [unique_protos[k] for k in k_values]

    fig, ax1 = plt.subplots(figsize=(12, 7))

    color = "tab:blue"
    ax1.set_xlabel("Number of Top Prototypes per Class (k)")
    ax1.set_ylabel("Validation Accuracy (%)", color=color)
    ax1.plot(k_values, accuracies, "o-", color=color, linewidth=2, markersize=8, label="Top-k Accuracy")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, which="both", ls="--", alpha=0.6)

    # --- NEW: Plot the full model's accuracy as a horizontal line ---
    if full_model_acc is not None:
        ax1.axhline(
            y=full_model_acc,
            color='green',
            linestyle=':',
            linewidth=2,
            label=f'Full Model Accuracy ({full_model_acc:.2f}%)'
        )
    ax1.legend(loc='lower right')
    # --- End of new section ---

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Total Unique Prototypes Used", color=color)
    ax2.plot(k_values, unique_counts, "s--", color=color, linewidth=2, markersize=6)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(0, total_protos * 1.05)

    fig.tight_layout(rect=(0, 0, 0.9, 0.95))
    plt.title("Model Performance vs. Number of Prototypes Used")

    plot_path = output_dir / "top_k_performance_vs_interpretability.png"
    plt.savefig(plot_path)
    logger.info(f"Saved performance plot to: {plot_path}")
    plt.close()



def analyze_prototypes(cfg: PipNetConfig, args: argparse.Namespace) -> None:
    """
    Main analysis function. It loads a trained PIP-Net model, performs the original
    local size analysis, and optionally runs the new top-k evaluation.

    Args:
        cfg (PipNetConfig): The configuration object for building the model.
        args (argparse.Namespace): The parsed command-line arguments.
    """
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    logger.info("Building and loading PIP-Net model from checkpoint...")
    try:
        model, head = build_pipnet_model(cfg, device)
        model.eval()
        head.eval()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load or build model: {e}", exc_info=True)
        return

    # --- Original Analysis: Weight Distribution Histogram ---
    logger.info("Visualizing weight distribution for a single class...")
    with torch.no_grad():
        class_0_weights = torch.relu(head.classifier.weight[0]).cpu().numpy()
        non_zero_weights = class_0_weights[class_0_weights > 0]

        if len(non_zero_weights) > 0:
            plt.figure(figsize=(12, 6))
            plt.hist(non_zero_weights, bins=100, log=True)
            plt.title(
                "Weight Distribution for Class 0 (Non-Zero Weights Only, Log Scale)"
            )
            plt.xlabel("Weight Value")
            plt.ylabel("Frequency (Log Scale)")
            plt.grid(True, which="both", ls="--")
            plot_path = output_dir / "class_0_weight_distribution.png"
            plt.savefig(plot_path)
            logger.info(f"Saved weight distribution plot to: {plot_path}")
            plt.close()
        else:
            logger.info("Class 0 has no positive weights to visualize.")

    # --- Original Analysis: Local Size Calculation ---
    logger.info(f"Calculating local sizes with threshold = {args.threshold}...")
    local_sizes = head.calculate_local_size(threshold=args.threshold)

    print("\n" + "=" * 60)
    print("      Prototype Local Size Analysis Report")
    print("=" * 60)
    for i, size in enumerate(local_sizes):
        print(f"  Class {i:3d}: {size.item():4d} significant prototypes")
    print("-" * 60)
    
    print_local_size_report(
        pipnet_checkpoint_path=args.pipnet_checkpoint_path,
        model_name=args.model_name,
        num_classes=args.num_classes,
        head=head,
        threshold=args.threshold,
        local_sizes=local_sizes,
    )

    if args.eval_top_k:
        print("\n" + "=" * 70)
        print(" " * 18 + "Top-K Prototype Evaluation")
        print("=" * 70)

        # 1. Parse k values from the command line string (e.g., "5,10,15")
        try:
            top_k_list = sorted([int(k.strip()) for k in args.eval_top_k.split(",")])
        except ValueError:
            logger.error(
                f"Invalid format for --eval-top-k. Please use comma-separated integers (e.g., '5,10,15')."
            )
            return

        # 2. Setup dataset and dataloader for validation
        if not args.dataset_name or not args.dataset_path:
            logger.error(
                "For top-k evaluation, --dataset-name and --dataset-path are required."
            )
            return

        logger.info(
            f"Loading validation data for '{args.dataset_name}' from '{args.dataset_path}'"
        )
        # Ensure fixed-size tensors for collation. CUB images are pre-cropped to birds,
        # but still variable-sized; enforce 224x224. For non-CUB, use 256->224 evaluation.
        is_cub = (args.dataset_name.lower() == "cub200")
        if is_cub:
            _, val_transform = get_default_image_transforms(
                resize_value=224,
                crop_value=224,
                is_precropped=True,
                horizontal_flip=None,
                autoaugment=False,
                random_erase=None,
            )
        else:
            _, val_transform = get_default_image_transforms(
                resize_value=256,
                crop_value=224,
                is_precropped=False,
                horizontal_flip=None,
                autoaugment=False,
                random_erase=None,
            )
        _, val_ds = get_dataset(
            name=args.dataset_name,
            train_transform=val_transform,  # pass a valid transform (train set unused here)
            val_transform=val_transform,
            path=args.dataset_path,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # 3. Run the core evaluation function
        results, unique_protos, avg_local_sizes = evaluate_with_top_k_prototypes(
            model=model,
            head=head,
            val_loader=val_loader,
            top_k_list=top_k_list,
            device=device,
            threshold=args.threshold,
        )

        # 4. Print results in a formatted table and generate the plot
        print("\n" + "-" * 70)
        print(f"{'K (Top Protos/Class)':<25} | {'Validation Accuracy (%)':<25} | {'Unique Protos Used':<20} | {'Avg Local Size/Class':<20}")
        print("-" * 70)
        key_order = ['Full'] + sorted([k for k in results.keys() if k != 'Full'])
        for k in key_order:
            k_str = str(k)
            print(f"{k_str:<25} | {results[k]:<25.2f} | {unique_protos[k]:<20} | {avg_local_sizes[k]:<20.2f}")
        print("=" * 70 + "\n")

        plot_results(results, unique_protos, head.P, output_dir)



def main():
    """
    Parses command-line arguments and launches the analysis.
    """
    parser = argparse.ArgumentParser(
        description="Analyze a trained PIP-Net model and evaluate prototype subsets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Arguments for model loading and basic analysis ---
    parser.add_argument(
        "--pipnet-checkpoint-path",
        type=Path,
        required=True,
        help="Path to the trained PIP-Net model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the backbone architecture, e.g., 'deit_small_patch16_224'.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        required=True,
        help="Number of classes in the dataset (for building the model).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Threshold for the initial local size analysis.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run analysis on.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_results",
        help="Directory to save plots and results.",
    )

    # --- New arguments for the Top-K Evaluation ---
    eval_group = parser.add_argument_group("Top-K Evaluation Options")
    eval_group.add_argument(
        "--eval-top-k",
        type=str,
        default=None,
        help="If provided, evaluates the model using only the top K prototypes per class. "
        "Provide a comma-separated list of integers, e.g., '5,10,15,20,50'.",
    )
    eval_group.add_argument(
        "--dataset-name",
        type=str,
        default="cub200",
        help="Name of the dataset to use for validation (e.g., 'cub200').",
    )
    eval_group.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to the root of the validation dataset. REQUIRED if --eval-top-k is used.",
    )
    eval_group.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for validation."
    )
    eval_group.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for data loading.",
    )

    args = parser.parse_args()

    if not args.pipnet_checkpoint_path.is_file():
        logger.error(f"Checkpoint file not found at: {args.pipnet_checkpoint_path}")
        return

    # Create a minimal configuration object required by the `build_pipnet_model` function.
    cfg = PipNetConfig()
    cfg.pipnet_checkpoint_path = str(args.pipnet_checkpoint_path)
    cfg.model.name = args.model_name
    cfg.model.global_pool = ""
    cfg.training.train_codebook = False  # Ensure we are in evaluation mode.
    cfg.dataset = BaseDatasetConfig(num_classes=args.num_classes)

    try:
        analyze_prototypes(cfg, args)
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()