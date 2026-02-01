import argparse
import logging
import os
import random
from pathlib import Path

import math
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from PIL import ImageDraw
import csv
from functools import lru_cache
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from src.config.pipnet_config import BaseDatasetConfig, PipNetConfig
from src.construct_model import construct_model, get_backbone
from src.datasets.construct_dataset import get_dataset
from src.datasets.transforms import (
    get_default_image_transforms,
    get_deit_transforms,
    get_raw_tensor_transforms,
)
from src.pipnet_utils import build_pipnet_model
import warnings

try:
    from pydantic.warnings import UnsupportedFieldAttributeWarning
except Exception:  # pragma: no cover - best-effort fallback
    UnsupportedFieldAttributeWarning = None

if UnsupportedFieldAttributeWarning is not None:
    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
else:
    warnings.filterwarnings(
        "ignore",
        message=".*UnsupportedFieldAttributeWarning.*",
    )


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _configure_logging(rank: int, is_distributed: bool) -> None:
    """Reduce logging noise on non-main ranks.

    Args:
        rank: Process rank.
        is_distributed: Whether distributed execution is enabled.
    """
    if not is_distributed or rank == 0:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.ERROR)
    for handler in root_logger.handlers:
        handler.setLevel(logging.ERROR)


class _IndexedDataset(Dataset):
    """Wrap a dataset to also return the sample index.

    Args:
        dataset: Base dataset instance to wrap.
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        if isinstance(sample, (tuple, list)):
            return (*sample, idx)
        return sample, idx


def _set_seed(seed: int) -> None:
    """Seed RNGs for reproducible sampling and model behavior.

    Args:
        seed: Random seed value.
    """
    # Deterministic sampling for reproducible visualizations
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_mean_std(
    use_raw: bool,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return normalization mean/std consistent with the selected transform pipeline.

    Args:
        use_raw: Whether raw (unnormalized) transforms are used.

    Returns:
        Tuple of mean and std (each a 3-tuple for RGB).
    """
    # Keep mean/std aligned with the transform pipeline
    if use_raw:
        return (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)

    return IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def _unnormalize_image(
    img_tensor: torch.Tensor,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    use_raw: bool,
) -> torch.Tensor:
    """Convert a normalized tensor back to [0, 1] RGB for visualization.

    Args:
        img_tensor: Normalized image tensor (CHW).
        mean: Normalization mean.
        std: Normalization std.
        use_raw: Whether raw (unnormalized) transforms are used.

    Returns:
        Unnormalized image tensor in [0, 1].
    """
    # Undo normalization so we can render the original image
    img = img_tensor.detach().cpu().clone()
    if use_raw:
        if img.max() > 1.0:
            img = img / 255.0
        return img.clamp(0.0, 1.0)

    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    img = img * std_t + mean_t
    return img.clamp(0.0, 1.0)


def _tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """Convert a CHW float tensor in [0,1] into a PIL RGB image.

    Args:
        img_tensor: Image tensor (CHW) in [0, 1].

    Returns:
        PIL image.
    """
    # Convert a CHW float tensor in [0,1] to PIL
    img = img_tensor.mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(img)


def _get_patch_params(
    image_size: int, fmap_size: int, patch_size: int
) -> tuple[int, int]:
    """Estimate patch size and stride to map feature indices to image space.

    Args:
        image_size: Input image dimension (height or width).
        fmap_size: Feature map dimension along the same axis.
        patch_size: Desired patch size in pixels.

    Returns:
        Tuple of (patch_size, stride).
    """
    # CNN mapping: approximate patch size and stride from feature map resolution
    patch_size = max(1, min(image_size, patch_size))
    if fmap_size <= 1:
        return patch_size, 0
    skip = round((image_size - patch_size) / (fmap_size - 1))
    return patch_size, max(0, skip)


def _get_img_coordinates(
    img_h: int,
    img_w: int,
    fmap_h: int,
    fmap_w: int,
    patch_h: int,
    patch_w: int,
    skip_h: int,
    skip_w: int,
    h_idx: int,
    w_idx: int,
) -> tuple[int, int, int, int]:
    """Map feature map indices back to image coordinates for cropping.

    Args:
        img_h: Image height in pixels.
        img_w: Image width in pixels.
        fmap_h: Feature map height.
        fmap_w: Feature map width.
        patch_h: Patch height in pixels.
        patch_w: Patch width in pixels.
        skip_h: Stride in pixels along height.
        skip_w: Stride in pixels along width.
        h_idx: Feature map row index.
        w_idx: Feature map column index.

    Returns:
        Crop box as (x_min, y_min, x_max, y_max).
    """
    # Ported from PIP-Net visualization utilities (CNN-focused)
    if fmap_h == 26 and fmap_w == 26:
        h_coor_min = max(0, (h_idx - 1) * skip_h + 4)
        if h_idx < fmap_h - 1:
            h_coor_max = h_coor_min + patch_h
        else:
            h_coor_min -= 4
            h_coor_max = h_coor_min + patch_h
        w_coor_min = max(0, (w_idx - 1) * skip_w + 4)
        if w_idx < fmap_w - 1:
            w_coor_max = w_coor_min + patch_w
        else:
            w_coor_min -= 4
            w_coor_max = w_coor_min + patch_w
    else:
        h_coor_min = h_idx * skip_h
        h_coor_max = min(img_h, h_idx * skip_h + patch_h)
        w_coor_min = w_idx * skip_w
        w_coor_max = min(img_w, w_idx * skip_w + patch_w)

    if h_idx == fmap_h - 1:
        h_coor_max = img_h
    if w_idx == fmap_w - 1:
        w_coor_max = img_w
    if h_coor_max == img_h:
        h_coor_min = img_h - patch_h
    if w_coor_max == img_w:
        w_coor_min = img_w - patch_w

    h_coor_min = max(0, min(img_h - 1, h_coor_min))
    w_coor_min = max(0, min(img_w - 1, w_coor_min))
    h_coor_max = max(h_coor_min + 1, min(img_h, h_coor_max))
    w_coor_max = max(w_coor_min + 1, min(img_w, w_coor_max))

    return w_coor_min, h_coor_min, w_coor_max, h_coor_max


def _make_patch_grid(
    patches: list[Image.Image],
    padding: int = 1,
    background: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Lay out patches horizontally into a single grid image.

    Args:
        patches: List of patch images (already resized).
        padding: Padding between patches.
        background: Background color as RGB tuple.

    Returns:
        Grid image.
    """
    if not patches:
        return Image.new("RGB", (1, 1), color=background)

    target_w, target_h = patches[0].size
    cols = len(patches)
    canvas_w = cols * target_w + (cols + 1) * padding
    canvas_h = target_h + 2 * padding
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=background)

    for i, patch in enumerate(patches):
        x = padding + i * (target_w + padding)
        y = padding
        canvas.paste(patch, (x, y))

    return canvas


def _update_topk_vectorized(
    topk_scores: torch.Tensor,
    topk_img_indices: torch.Tensor,
    topk_flat_indices: torch.Tensor,
    max_sim: torch.Tensor,
    max_idx: torch.Tensor,
    img_indices: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized top-k update across prototypes.

    Args:
        topk_scores: Tensor (P, k) of current best scores.
        topk_img_indices: Tensor (P, k) of dataset indices.
        topk_flat_indices: Tensor (P, k) of flat feature-map indices.
        max_sim: Max similarity per (batch, prototype).
        max_idx: Flat index of max similarity per (batch, prototype).
        img_indices: Dataset indices for the batch.
        k: Number of entries to keep per prototype.

    Returns:
        Updated (scores, img_indices, flat_indices) tensors.
    """
    batch_size, num_prototypes = max_sim.shape
    batch_scores = max_sim.transpose(0, 1)  # (P, B)
    batch_flat_indices = max_idx.transpose(0, 1)  # (P, B)
    batch_img_indices = img_indices.view(1, batch_size).expand(
        num_prototypes, batch_size
    )

    combined_scores = torch.cat([topk_scores, batch_scores], dim=1)
    combined_img_indices = torch.cat([topk_img_indices, batch_img_indices], dim=1)
    combined_flat_indices = torch.cat([topk_flat_indices, batch_flat_indices], dim=1)

    topk_scores, topk_positions = torch.topk(combined_scores, k=k, dim=1)
    topk_img_indices = torch.gather(combined_img_indices, 1, topk_positions)
    topk_flat_indices = torch.gather(combined_flat_indices, 1, topk_positions)

    return topk_scores, topk_img_indices, topk_flat_indices


def _load_train_loader(
    dataset_name: str,
    dataset_path: str,
    val_transform,
    batch_size: int,
    num_workers: int,
    is_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[Dataset, DataLoader]:
    """Load the training dataset and build its dataloader.

    Args:
        dataset_name: Dataset name (e.g., "cub200").
        dataset_path: Path to the dataset root.
        val_transform: Transform pipeline used for evaluation/visualization.
        batch_size: Batch size for the training loader.
        num_workers: Number of dataloader workers.

    Returns:
        Tuple of (train_dataset, train_loader).
    """
    train_ds, _ = get_dataset(
        name=dataset_name,
        train_transform=val_transform,
        val_transform=val_transform,
        path=dataset_path,
    )

    indexed_train = _IndexedDataset(train_ds)
    sampler = None
    if is_distributed and world_size > 1:
        sampler = DistributedSampler(
            indexed_train,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
    loader = DataLoader(
        indexed_train,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_ds, loader


def _init_distributed(rank: int, world_size: int) -> tuple[int, int, torch.device]:
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    init_method = os.environ.get("INIT_METHOD")
    if not init_method:
        raise ValueError(
            "INIT_METHOD environment variable must be set for distributed runs."
        )

    local_rank = rank
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )
    return rank, world_size, device


def _unpack_batch(batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Return tensors (xs, idxs) from a batch, ignoring labels if present.

    Args:
        batch: Batch from the dataloader.

    Returns:
        Tuple of (inputs, indices).
    """
    if isinstance(batch, (tuple, list)):
        if len(batch) == 3:
            inputs, _, indices = batch
        elif len(batch) == 2:
            inputs, indices = batch
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")
    else:
        raise ValueError("Unexpected batch format (expected tuple or list)")
    return inputs, indices


def _extract_image_tensor(sample) -> torch.Tensor:
    """Extract the image tensor from a dataset sample.

    Args:
        sample: Dataset sample (image or (image, label, ...)).

    Returns:
        Image tensor (CHW).
    """
    if isinstance(sample, (tuple, list)):
        img_tensor = sample[0]
    else:
        img_tensor = sample

    if img_tensor.ndim == 4:
        img_tensor = img_tensor.squeeze(0)
    return img_tensor


def _compute_similarity_topk(
    model,
    loader: DataLoader,
    device: torch.device,
    k: int,
) -> tuple[list[list[tuple[float, int, int]]], int, int]:
    """Scan the training set and return per-prototype top-k (score, img, flat_idx).

    Args:
        model: ProtoQuant/PIP-Net model.
        loader: Training dataloader with indices.
        device: Device for inference.
        k: Number of top entries per prototype.

    Returns:
        Tuple of (topk_entries, feature_map_h, feature_map_w).
    """
    topk_scores: torch.Tensor | None = None
    topk_img_indices: torch.Tensor | None = None
    topk_flat_indices: torch.Tensor | None = None
    fmap_h = None
    fmap_w = None
    log_every = 10

    for batch_idx, batch in enumerate(loader, start=1):
        inputs, indices = _unpack_batch(batch)
        inputs = inputs.to(device)
        indices = indices.to(device)

        out = model(inputs, return_similarity_map=True)
        similarity_map = out.similarity_map
        if similarity_map is None:
            logger.warning("No similarity map returned for a batch; skipping.")
            continue

        temperature = float(getattr(model, "temperature", 1.0))
        softmax_map = F.softmax(similarity_map / temperature, dim=1)

        batch_size, num_prototypes, fmap_height, fmap_width = softmax_map.shape
        if topk_scores is None:
            topk_scores = torch.full(
                (num_prototypes, k),
                float("-inf"),
                device=softmax_map.device,
            )
            topk_img_indices = torch.full(
                (num_prototypes, k),
                -1,
                dtype=torch.long,
                device=softmax_map.device,
            )
            topk_flat_indices = torch.full(
                (num_prototypes, k),
                -1,
                dtype=torch.long,
                device=softmax_map.device,
            )
            fmap_h = fmap_height
            fmap_w = fmap_width

        flat = softmax_map.view(batch_size, num_prototypes, -1)
        max_sim, max_idx = torch.max(flat, dim=2)
        topk_scores, topk_img_indices, topk_flat_indices = _update_topk_vectorized(
            topk_scores,
            topk_img_indices,
            topk_flat_indices,
            max_sim,
            max_idx,
            indices,
            k=k,
        )

        if batch_idx == 1 or batch_idx % log_every == 0:
            logger.info(
                "Scan batch %s | device=%s | batch=%s | fmap=%sx%s | protos=%s",
                batch_idx,
                inputs.device,
                batch_size,
                fmap_height,
                fmap_width,
                num_prototypes,
            )

    if (
        topk_scores is None
        or topk_img_indices is None
        or topk_flat_indices is None
        or fmap_h is None
        or fmap_w is None
    ):
        raise RuntimeError("No similarity maps collected. Check dataset and model.")

    topk: list[list[tuple[float, int, int]]] = []
    topk_scores_cpu = topk_scores.detach().cpu()
    topk_img_cpu = topk_img_indices.detach().cpu()
    topk_flat_cpu = topk_flat_indices.detach().cpu()
    for proto_idx in range(topk_scores_cpu.shape[0]):
        entries: list[tuple[float, int, int]] = []
        for rank_idx in range(topk_scores_cpu.shape[1]):
            score = float(topk_scores_cpu[proto_idx, rank_idx].item())
            img_idx = int(topk_img_cpu[proto_idx, rank_idx].item())
            flat_idx = int(topk_flat_cpu[proto_idx, rank_idx].item())
            if img_idx < 0:
                continue
            entries.append((score, img_idx, flat_idx))
        topk.append(entries)

    return topk, fmap_h, fmap_w


def _extract_patch_from_image(
    img_tensor: torch.Tensor,
    flat_idx: int,
    fmap_h: int,
    fmap_w: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    patch_size: int,
    use_raw_transforms: bool,
) -> Image.Image:
    """Crop the patch at a feature-map location from a single image tensor.

    Args:
        img_tensor: Image tensor (CHW).
        flat_idx: Flattened feature-map index for the peak location.
        fmap_h: Feature map height.
        fmap_w: Feature map width.
        mean: Normalization mean.
        std: Normalization std.
        patch_size: Patch size used for mapping to image space.
        use_raw_transforms: Whether raw (unnormalized) transforms are used.

    Returns:
        Cropped patch as a PIL image.
    """
    img_vis = _unnormalize_image(img_tensor, mean, std, use_raw_transforms)
    img_pil_raw = _tensor_to_pil(img_vis)
    width, height = img_pil_raw.size
    patch_h, skip_h = _get_patch_params(height, fmap_h, patch_size)
    patch_w, skip_w = _get_patch_params(width, fmap_w, patch_size)

    h_idx = int(flat_idx // fmap_w)
    w_idx = int(flat_idx % fmap_w)

    x_min, y_min, x_max, y_max = _get_img_coordinates(
        img_h=height,
        img_w=width,
        fmap_h=fmap_h,
        fmap_w=fmap_w,
        patch_h=patch_h,
        patch_w=patch_w,
        skip_h=skip_h,
        skip_w=skip_w,
        h_idx=h_idx,
        w_idx=w_idx,
    )

    if x_max <= x_min or y_max <= y_min:
        logger.warning(
            "Invalid crop (flat=%s). Using center patch.",
            flat_idx,
        )
        cx = width // 2
        cy = height // 2
        x_min = max(0, cx - patch_w // 2)
        y_min = max(0, cy - patch_h // 2)
        x_max = min(width, x_min + patch_w)
        y_max = min(height, y_min + patch_h)

    return img_pil_raw.crop((x_min, y_min, x_max, y_max)).resize(
        (patch_w, patch_h), Image.Resampling.BICUBIC
    )


def _get_bbox_from_flat_idx(
    img_tensor: torch.Tensor,
    flat_idx: int,
    fmap_h: int,
    fmap_w: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    patch_size: int,
    use_raw_transforms: bool,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Return the unnormalized PIL image and bbox coordinates for a feature index.

    Args:
        img_tensor: Image tensor (CHW).
        flat_idx: Flattened feature-map index.
        fmap_h: Feature map height.
        fmap_w: Feature map width.
        mean: Normalization mean.
        std: Normalization std.
        patch_size: Patch size used for mapping to image space.
        use_raw_transforms: Whether raw (unnormalized) transforms are used.

    Returns:
        Tuple of (PIL image, (x_min, y_min, x_max, y_max)).
    """
    img_vis = _unnormalize_image(img_tensor, mean, std, use_raw_transforms)
    img_pil_raw = _tensor_to_pil(img_vis)
    width, height = img_pil_raw.size
    patch_h, skip_h = _get_patch_params(height, fmap_h, patch_size)
    patch_w, skip_w = _get_patch_params(width, fmap_w, patch_size)

    h_idx = int(flat_idx // fmap_w)
    w_idx = int(flat_idx % fmap_w)

    x_min, y_min, x_max, y_max = _get_img_coordinates(
        img_h=height,
        img_w=width,
        fmap_h=fmap_h,
        fmap_w=fmap_w,
        patch_h=patch_h,
        patch_w=patch_w,
        skip_h=skip_h,
        skip_w=skip_w,
        h_idx=h_idx,
        w_idx=w_idx,
    )
    return img_pil_raw, (x_min, y_min, x_max, y_max)


def _get_bbox_coords(
    img_w: int,
    img_h: int,
    flat_idx: int,
    fmap_h: int,
    fmap_w: int,
    patch_size: int,
) -> tuple[int, int, int, int]:
    """Return bbox coords for a feature index given image size."""
    patch_h, skip_h = _get_patch_params(img_h, fmap_h, patch_size)
    patch_w, skip_w = _get_patch_params(img_w, fmap_w, patch_size)

    h_idx = int(flat_idx // fmap_w)
    w_idx = int(flat_idx % fmap_w)

    return _get_img_coordinates(
        img_h=img_h,
        img_w=img_w,
        fmap_h=fmap_h,
        fmap_w=fmap_w,
        patch_h=patch_h,
        patch_w=patch_w,
        skip_h=skip_h,
        skip_w=skip_w,
        h_idx=h_idx,
        w_idx=w_idx,
    )


def _save_prototype_grid(
    proto_idx: int,
    patches: list[Image.Image],
    output_dir: Path,
    title: str | None = None,
) -> None:
    """Save a single horizontal grid PNG per prototype.

    Args:
        proto_idx: Prototype index.
        patches: Patch images ordered by rank.
        output_dir: Output directory for grid images.
    """
    if not patches:
        return

    grid = _make_patch_grid(patches)
    if title:
        grid = _add_title_to_grid(grid, title)
    grid_path = output_dir / f"prototype_{proto_idx:05d}.png"
    grid.save(grid_path)
    logger.info("Saved %s", grid_path)


def _add_title_to_grid(
    grid: Image.Image,
    title: str,
    padding: int = 6,
    background: tuple[int, int, int] = (255, 255, 255),
    text_color: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Return a new image with a title rendered above the grid."""
    draw = ImageDraw.Draw(grid)
    try:
        text_bbox = draw.textbbox((0, 0), title)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
    except AttributeError:
        text_w, text_h = draw.textsize(title)

    title_h = text_h + 2 * padding
    new_w = max(grid.width, text_w + 2 * padding)
    new_h = grid.height + title_h
    canvas = Image.new("RGB", (new_w, new_h), color=background)
    canvas_draw = ImageDraw.Draw(canvas)
    canvas_draw.text((padding, padding), title, fill=text_color)
    x = (new_w - grid.width) // 2
    canvas.paste(grid, (x, title_h))
    return canvas


def _make_image_grid(
    images: list[Image.Image],
    columns: int,
    padding: int = 6,
    background: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Arrange images in a grid with a fixed number of columns."""
    if not images:
        return Image.new("RGB", (1, 1), color=background)

    columns = max(1, columns)
    cell_w, cell_h = images[0].size
    rows = (len(images) + columns - 1) // columns
    canvas_w = columns * cell_w + (columns + 1) * padding
    canvas_h = rows * cell_h + (rows + 1) * padding
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=background)

    for idx, img in enumerate(images):
        row = idx // columns
        col = idx % columns
        x = padding + col * (cell_w + padding)
        y = padding + row * (cell_h + padding)
        canvas.paste(img, (x, y))

    return canvas


@lru_cache(maxsize=4)
def _load_cub_class_names(cub_root: str) -> list[str]:
    classes_path = Path(cub_root) / "classes.txt"
    if not classes_path.is_file():
        return []
    names: list[str] = []
    with classes_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                names.append(parts[1])
    return names


def _get_topk_prototypes_per_class(
    model,
    k: int,
) -> list[list[int]]:
    """Return top-k prototype indices per class based on classifier weights."""
    weights = model.classifier.weight.detach().cpu()
    num_classes, num_prototypes = weights.shape
    k = max(1, min(k, num_prototypes))
    topk_per_class: list[list[int]] = []
    for class_idx in range(num_classes):
        class_weights = weights[class_idx]
        _, indices = torch.topk(class_weights, k=k)
        topk_per_class.append(indices.tolist())
    return topk_per_class


def _get_class_name(dataset: Dataset, class_idx: int) -> str:
    """Return class name for CUB200 or ImageNet1K datasets."""
    try:
        classes = dataset.classes
    except AttributeError:
        classes = None
    if isinstance(classes, (list, tuple)) and 0 <= class_idx < len(classes):
        return str(classes[class_idx])

    try:
        root = dataset.root
    except AttributeError:
        root = None
    if root:
        names = _load_cub_class_names(str(root))
        if 0 <= class_idx < len(names):
            return names[class_idx]

    return str(class_idx)


def _draw_bbox(
    img: Image.Image,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int] = (255, 0, 0),
    width: int = 3,
) -> Image.Image:
    """Return a copy of the image with a bounding box drawn."""
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    draw.rectangle(bbox, outline=color, width=width)
    return img_copy


def _draw_bboxes(
    img: Image.Image,
    bboxes: list[tuple[int, int, int, int]],
    colors: list[tuple[int, int, int]],
    width: int = 3,
) -> Image.Image:
    """Return a copy of the image with multiple bboxes drawn."""
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        draw.rectangle(bbox, outline=color, width=width)
    return img_copy


def _build_match_grid(
    base_img: Image.Image,
    base_bboxes: list[tuple[int, int, int, int]],
    base_colors: list[tuple[int, int, int]],
    match_rows: list[list[tuple[Image.Image, tuple[int, int, int, int]]]],
    columns: int,
    padding: int = 6,
    background: tuple[int, int, int] = (255, 255, 255),
    bbox_width: int = 3,
) -> Image.Image:
    """Build a grid where each row is the base image plus nearest matches."""
    if not base_bboxes:
        return base_img.copy()

    target_w, target_h = base_img.size
    total_cols = 1 + columns
    total_rows = len(base_bboxes)

    canvas_w = total_cols * target_w + (total_cols + 1) * padding
    canvas_h = total_rows * target_h + (total_rows + 1) * padding
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=background)

    for row_idx, bbox in enumerate(base_bboxes):
        color = base_colors[row_idx % len(base_colors)]
        row_y = padding + row_idx * (target_h + padding)

        base_with_bbox = _draw_bbox(base_img, bbox, color=color, width=bbox_width)
        base_with_bbox = base_with_bbox.resize(
            (target_w, target_h), Image.Resampling.BICUBIC
        )
        canvas.paste(base_with_bbox, (padding, row_y))

        matches = match_rows[row_idx] if row_idx < len(match_rows) else []
        for col_idx in range(columns):
            x = padding + (col_idx + 1) * (target_w + padding)
            if col_idx < len(matches):
                match_img, match_bbox = matches[col_idx]
                match_img = match_img.resize(
                    (target_w, target_h), Image.Resampling.BICUBIC
                )
                match_with_bbox = _draw_bbox(
                    match_img, match_bbox, color=color, width=bbox_width
                )
                canvas.paste(match_with_bbox, (x, row_y))
            else:
                blank = Image.new("RGB", (target_w, target_h), color=background)
                canvas.paste(blank, (x, row_y))

    return canvas


def _build_composite_image(
    base_img: Image.Image,
    bboxes: list[tuple[int, int, int, int]],
    grid_images: list[Image.Image],
    colors: list[tuple[int, int, int]],
    panel_gap: int = 12,
    panel_padding: int = 8,
    grid_padding: int = 6,
    grid_border: int = 3,
    link_lines: bool = False,
) -> Image.Image:
    """Create a composite image with bboxes and corresponding grids on the right."""
    img_w, img_h = base_img.size
    num_items = min(len(bboxes), len(grid_images))
    if num_items == 0:
        return base_img.copy()

    grid_sizes = [grid_images[i].size for i in range(num_items)]
    max_grid_w = max(w for w, _ in grid_sizes)
    total_grid_h = sum(h for _, h in grid_sizes)
    panel_w = max_grid_w + 2 * (panel_padding + grid_border)
    panel_h = total_grid_h + (num_items + 1) * grid_padding + 2 * panel_padding

    composite_w = img_w + panel_gap + panel_w
    composite_h = max(img_h, panel_h)
    composite = Image.new("RGB", (composite_w, composite_h), color=(255, 255, 255))

    composite.paste(base_img, (0, 0))
    panel_x = img_w + panel_gap
    panel_y = (composite_h - panel_h) // 2

    draw = ImageDraw.Draw(composite)

    y_cursor = panel_y + panel_padding + grid_padding
    for i in range(num_items):
        grid_img = grid_images[i]
        grid_w, grid_h = grid_img.size
        color = colors[i % len(colors)]

        x = panel_x + panel_padding + grid_border
        y = y_cursor

        composite.paste(grid_img, (x, y))

        border_box = (
            x - grid_border,
            y - grid_border,
            x + grid_w + grid_border,
            y + grid_h + grid_border,
        )
        draw.rectangle(border_box, outline=color, width=grid_border)

        if link_lines:
            bbox = bboxes[i]
            bbox_cx = (bbox[0] + bbox[2]) // 2
            bbox_cy = (bbox[1] + bbox[3]) // 2
            grid_cx = x + grid_w // 2
            grid_cy = y + grid_h // 2
            draw.line([(bbox_cx, bbox_cy), (grid_cx, grid_cy)], fill=color, width=2)

        y_cursor += grid_h + grid_padding

    return composite


@torch.no_grad()
def visualize_protoquant(
    args: argparse.Namespace, rank: int = 0, world_size: int = 1
) -> None:
    """Generate top-k nearest training patches for each prototype and save grids.

    Args:
        args: CLI arguments.
    """
    is_distributed = args.distributed and world_size > 1
    if is_distributed:
        rank, world_size, device = _init_distributed(rank, world_size)
    else:
        device = torch.device(args.device)
    _configure_logging(rank, is_distributed)
    output_dir = Path(args.save_path)
    composites_dir = None
    if not is_distributed or rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        grid_dir = output_dir / "grids"
        grid_dir.mkdir(parents=True, exist_ok=True)
        if args.save_bboxes:
            boxes_dir = output_dir / "boxes"
            boxes_dir.mkdir(parents=True, exist_ok=True)
        if args.save_composite:
            composites_dir = output_dir / "composites"
            composites_dir.mkdir(parents=True, exist_ok=True)
    else:
        grid_dir = None
        boxes_dir = None
        composites_dir = None

    _set_seed(args.seed)
    logger.info(
        "Starting scan on device=%s | distributed=%s | rank=%s/%s",
        device,
        is_distributed,
        rank,
        world_size,
    )

    # Match evaluation transforms to training config
    is_cub = args.dataset_name.lower() == "cub200"
    if args.use_raw_transforms:
        _, val_transform = get_raw_tensor_transforms(resize=args.image_size)
    elif args.use_deit_transforms:
        _, val_transform = get_deit_transforms(is_precropped=is_cub)
    else:
        _, val_transform = get_default_image_transforms(
            autoaugment=False,
            resize_value=args.image_size if is_cub else 256,
            crop_value=None if is_cub else 224,
            random_erase=None,
            horizontal_flip=None,
            is_precropped=is_cub,
        )

    # Load the training split (prototype search set)
    train_ds, loader = _load_train_loader(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        val_transform=val_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_distributed=is_distributed,
        rank=rank,
        world_size=world_size,
    )

    # Minimal config for construct_model
    cfg = PipNetConfig()
    cfg.pipnet_checkpoint_path = str(args.pipnet_checkpoint_path)
    cfg.model.name = args.model_name
    cfg.model.global_pool = ""
    cfg.training.train_codebook = False
    cfg.dataset = BaseDatasetConfig(num_classes=args.num_classes)

    base_model = construct_model(cfg, device)
    backbone = get_backbone(base_model)

    # Load ProtoQuant with the codebook and classifier from checkpoint
    model = build_pipnet_model(
        backbone=backbone,
        num_classes=args.num_classes,
        device=device,
        pipnet_checkpoint_path=str(args.pipnet_checkpoint_path),
        train_codebook=False,
    )
    model.eval()

    if args.limit_prototypes > 0:
        remaining = model.limit_prototypes(k=args.limit_prototypes)
        logger.info(
            "Limited prototypes to top-%s per class. Remaining: %s",
            args.limit_prototypes,
            remaining,
        )

    proto_id_map = list(range(int(getattr(model, "num_prototypes", 0))))
    if args.prune_inactive_prototypes:
        with torch.no_grad():
            weights = torch.relu(model.classifier.weight.detach())
            active_mask = (weights > float(args.prune_min_weight)).any(dim=0)
            active_indices = (
                active_mask.nonzero(as_tuple=False).flatten().cpu().tolist()
            )
        proto_id_map = active_indices
        remaining = model.prune_inactive_prototypes(min_weight=args.prune_min_weight)
        logger.info("Pruned inactive prototypes. Remaining: %s", remaining)

    mean, std = _get_mean_std(args.use_raw_transforms)

    try:
        local_topk, fmap_h, fmap_w = _compute_similarity_topk(
            model=model,
            loader=loader,
            device=device,
            k=args.num_prototypes,
        )
    except RuntimeError as exc:
        logger.error(str(exc))
        if is_distributed:
            dist.destroy_process_group()
        return

    if is_distributed and world_size > 1:
        gathered: list[list[list[tuple[float, int, int]]]] | None = None
        if rank == 0:
            gathered = [None for _ in range(world_size)]  # type: ignore[list-item]
        dist.gather_object(local_topk, gathered, dst=0)

        if rank != 0:
            dist.destroy_process_group()
            return

        # Merge per-rank top-k lists on rank 0
        assert gathered is not None
        topk: list[list[tuple[float, int, int]]] = [[] for _ in range(len(gathered[0]))]
        for rank_topk in gathered:
            for p_idx, entries in enumerate(rank_topk):
                topk[p_idx].extend(entries)

        # Keep only global top-k per prototype
        for p_idx in range(len(topk)):
            topk[p_idx] = sorted(topk[p_idx], key=lambda t: t[0], reverse=True)[
                : args.num_prototypes
            ]
    else:
        topk = local_topk

    max_prototypes = min(args.max_prototypes, len(topk))
    if args.per_class_topk:
        per_class_dir = output_dir / "per_class"
        per_class_dir.mkdir(parents=True, exist_ok=True)
        topk_per_class = _get_topk_prototypes_per_class(
            model=model,
            k=args.per_class_topk_k,
        )
        for class_idx, proto_indices in enumerate(topk_per_class):
            class_dir = per_class_dir / f"class_{class_idx:03d}"
            class_dir.mkdir(parents=True, exist_ok=True)
            class_name = _get_class_name(train_ds, class_idx)
            logger.info(
                "Class %s (%s) | top-%s prototypes: %s",
                class_idx,
                class_name,
                args.per_class_topk_k,
                proto_indices,
            )
            proto_grids: list[Image.Image] = []
            for proto_index in proto_indices:
                if proto_index >= len(topk):
                    continue
                entries = topk[proto_index]
                if not entries:
                    continue
                entries = sorted(entries, key=lambda t: t[0], reverse=True)
                if args.min_similarity > 0.0:
                    entries = [
                        entry for entry in entries if entry[0] >= args.min_similarity
                    ]
                    if not entries:
                        continue

                patches: list[Image.Image] = []
                for _, (score, img_idx, flat_idx) in enumerate(
                    entries[: args.num_prototypes], start=1
                ):
                    img_tensor = _extract_image_tensor(train_ds[img_idx])
                    patch = _extract_patch_from_image(
                        img_tensor=img_tensor,
                        flat_idx=flat_idx,
                        fmap_h=fmap_h,
                        fmap_w=fmap_w,
                        mean=mean,
                        std=std,
                        patch_size=args.patch_size,
                        use_raw_transforms=args.use_raw_transforms,
                    )
                    patches.append(patch)
                if patches:
                    proto_grids.append(_make_patch_grid(patches))

                if proto_grids:
                    if args.per_class_grid_columns and args.per_class_grid_columns > 0:
                        columns = min(args.per_class_grid_columns, len(proto_grids))
                    else:
                        columns = max(1, math.isqrt(len(proto_grids)))
                        if columns * columns < len(proto_grids):
                            columns += 1
                class_grid = _make_image_grid(
                    proto_grids,
                    columns=columns,
                )
                class_grid = _add_title_to_grid(
                    class_grid,
                    f"{class_idx}: {class_name}",
                )
                class_path = class_dir / f"class_{class_idx:03d}.png"
                class_grid.save(class_path)
                logger.info("Saved %s", class_path)
    else:
        for proto_idx in range(max_prototypes):
            entries = topk[proto_idx]
            logger.info("Grid %s/%s", proto_idx + 1, max_prototypes)
            if not entries:
                continue
            entries = sorted(entries, key=lambda t: t[0], reverse=True)
            if args.min_similarity > 0.0:
                entries = [
                    entry for entry in entries if entry[0] >= args.min_similarity
                ]
                if not entries:
                    continue

            patches: list[Image.Image] = []
            for _, (score, img_idx, flat_idx) in enumerate(
                entries[: args.num_prototypes], start=1
            ):
                img_tensor = _extract_image_tensor(train_ds[img_idx])
                patch = _extract_patch_from_image(
                    img_tensor=img_tensor,
                    flat_idx=flat_idx,
                    fmap_h=fmap_h,
                    fmap_w=fmap_w,
                    mean=mean,
                    std=std,
                    patch_size=args.patch_size,
                    use_raw_transforms=args.use_raw_transforms,
                )
                patches.append(patch)

            _save_prototype_grid(
                proto_idx=proto_idx,
                patches=patches,
                output_dir=grid_dir if grid_dir is not None else output_dir,
            )

    if (
        (not is_distributed or rank == 0)
        and args.save_bboxes
        and args.max_bbox_images > 0
    ):
        saved_images = 0
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 128, 255),
            (255, 128, 0),
            (255, 0, 255),
        ]
        color_names = [
            "red",
            "green",
            "blue",
            "orange",
            "magenta",
        ]
        rng = random.Random(args.seed)
        all_indices = list(range(len(train_ds)))
        rng.shuffle(all_indices)
        sample_indices = all_indices[: args.max_bbox_images]
        bbox_dataset = Subset(_IndexedDataset(train_ds), sample_indices)
        bbox_loader = DataLoader(
            bbox_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        for batch in bbox_loader:
            inputs, indices = _unpack_batch(batch)
            inputs = inputs.to(device)
            indices = indices.to(device)

            out = model(inputs, return_similarity_map=True)
            similarity_map = out.similarity_map
            if similarity_map is None:
                continue

            temperature = float(getattr(model, "temperature", 1.0))
            softmax_map = F.softmax(similarity_map / temperature, dim=1)
            batch_size, num_prototypes, _, _ = softmax_map.shape
            if num_prototypes == 0:
                break

            flat = softmax_map.view(batch_size, num_prototypes, -1)
            max_sim, max_idx = torch.max(flat, dim=2)  # (B, P)

            pooled = max_sim  # (B, P)
            class_logits = model.classifier(pooled)
            class_pred = torch.argmax(class_logits, dim=1)

            for i in range(batch_size):
                if saved_images >= args.max_bbox_images:
                    break

                img_idx = int(indices[i].item())
                img_tensor = _extract_image_tensor(train_ds[img_idx])
                img_vis = _unnormalize_image(
                    img_tensor, mean, std, args.use_raw_transforms
                )
                img_pil = _tensor_to_pil(img_vis)

                topk_k = min(args.bboxes_per_image, num_prototypes)
                if topk_k < 1:
                    continue

                pred_class = int(class_pred[i].item())
                class_score = float(class_logits[i, pred_class].item())
                class_weights = torch.relu(model.classifier.weight[pred_class])
                proto_scores = class_weights * pooled[i]
                if args.min_similarity > 0.0:
                    keep_mask = proto_scores >= args.min_similarity
                    if not torch.any(keep_mask):
                        continue
                    kept_scores = proto_scores[keep_mask]
                    kept_indices = torch.nonzero(keep_mask, as_tuple=False).flatten()
                    topk_k = min(topk_k, kept_scores.numel())
                    scores, kept_pos = torch.topk(kept_scores, k=topk_k)
                    proto_indices = kept_indices[kept_pos]
                else:
                    scores, proto_indices = torch.topk(proto_scores, k=topk_k)
                flat_indices = max_idx[i].gather(0, proto_indices)

                bboxes: list[tuple[int, int, int, int]] = []
                rows: list[list[str | int | float]] = []
                grid_images: list[Image.Image] = []
                match_rows: list[
                    list[tuple[Image.Image, tuple[int, int, int, int]]]
                ] = []

                for rank_idx, (score, p_idx, f_idx) in enumerate(
                    zip(scores, proto_indices, flat_indices), start=1
                ):
                    color_idx = (rank_idx - 1) % len(colors)
                    color_name = color_names[color_idx]
                    proto_index = int(p_idx.item())
                    proto_id = (
                        int(proto_id_map[proto_index])
                        if proto_index < len(proto_id_map)
                        else proto_index
                    )
                    flat_idx = int(f_idx.item())
                    bbox = _get_bbox_coords(
                        img_w=img_pil.size[0],
                        img_h=img_pil.size[1],
                        flat_idx=flat_idx,
                        fmap_h=fmap_h,
                        fmap_w=fmap_w,
                        patch_size=args.patch_size,
                    )
                    bboxes.append(bbox)
                    rows.append(
                        [
                            img_idx,
                            pred_class,
                            class_score,
                            rank_idx,
                            proto_index,
                            proto_id,
                            float(score),
                            flat_idx,
                            bbox[0],
                            bbox[1],
                            bbox[2],
                            bbox[3],
                            color_name,
                            f"grids/prototype_{proto_index:05d}.png",
                        ]
                    )

                    if args.save_composite and grid_dir is not None:
                        grid_path = grid_dir / f"prototype_{proto_index:05d}.png"
                        if grid_path.is_file():
                            with Image.open(grid_path) as grid_img:
                                grid_images.append(grid_img.convert("RGB").copy())
                        else:
                            grid_images.append(
                                Image.new("RGB", (1, 1), color=(0, 0, 0))
                            )

                    if args.save_match_grid:
                        proto_entries = (
                            topk[proto_index] if proto_index < len(topk) else []
                        )
                        proto_entries = sorted(
                            proto_entries, key=lambda t: t[0], reverse=True
                        )
                        row_matches: list[
                            tuple[Image.Image, tuple[int, int, int, int]]
                        ] = []
                        for entry in proto_entries[: args.match_grid_columns]:
                            _, match_img_idx, match_flat_idx = entry
                            match_tensor = _extract_image_tensor(
                                train_ds[match_img_idx]
                            )
                            match_vis = _unnormalize_image(
                                match_tensor, mean, std, args.use_raw_transforms
                            )
                            match_img = _tensor_to_pil(match_vis)
                            match_bbox = _get_bbox_coords(
                                img_w=match_img.size[0],
                                img_h=match_img.size[1],
                                flat_idx=match_flat_idx,
                                fmap_h=fmap_h,
                                fmap_w=fmap_w,
                                patch_size=args.patch_size,
                            )
                            row_matches.append((match_img, match_bbox))
                        match_rows.append(row_matches)

                boxed = _draw_bboxes(img_pil, bboxes, colors)
                image_name = f"image_{img_idx:06d}.png"
                image_path = output_dir / "boxes" / image_name
                if args.save_match_grid:
                    match_grid = _build_match_grid(
                        base_img=img_pil,
                        base_bboxes=bboxes,
                        base_colors=colors,
                        match_rows=match_rows,
                        columns=args.match_grid_columns,
                        padding=args.match_grid_padding,
                    )
                    match_grid.save(image_path)
                else:
                    boxed.save(image_path)

                if args.save_composite and composites_dir is not None:
                    composite = _build_composite_image(
                        base_img=boxed,
                        bboxes=bboxes,
                        grid_images=grid_images,
                        colors=colors,
                        link_lines=args.composite_link_lines,
                    )
                    composite_path = composites_dir / image_name
                    composite.save(composite_path)

                csv_path = output_dir / "boxes" / f"image_{img_idx:06d}.csv"
                with csv_path.open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "image_index",
                            "predicted_class",
                            "predicted_class_score",
                            "rank",
                            "prototype_index",
                            "prototype_id",
                            "similarity_score",
                            "flat_index",
                            "x_min",
                            "y_min",
                            "x_max",
                            "y_max",
                            "bbox_color_name",
                            "grid_path",
                        ]
                    )
                    writer.writerows(rows)

                logger.info("Saved %s", image_path)
                saved_images += 1

            if saved_images >= args.max_bbox_images:
                break

    if is_distributed:
        dist.destroy_process_group()


def _distributed_worker(rank: int, args: argparse.Namespace) -> None:
    visualize_protoquant(args, rank=rank, world_size=args.world_size)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize top-k nearest training patches per prototype.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--pipnet-checkpoint-path",
        type=Path,
        required=True,
        help="Path to the trained ProtoQuant checkpoint (.pth).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Backbone model name (e.g., 'deit_small_patch16_224').",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        required=True,
        help="Number of classes used to build the model.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Dataset name (e.g., 'cub200').",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the dataset root.",
    )
    parser.add_argument(
        "--num-prototypes",
        type=int,
        default=10,
        help="Number of nearest training patches per prototype to save.",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.01,
        help="Minimum similarity score to include prototypes in grids/bboxes (0 disables).",
    )
    parser.add_argument(
        "--max-prototypes",
        type=int,
        default=100000,
        help="Maximum number of prototypes to visualize.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for scanning the training set.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of dataloader workers for scanning the training set.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size used for visualization and patch mapping.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=32,
        help="Patch size used to map feature locations back to image space (CNN).",
    )
    parser.add_argument(
        "--limit-prototypes",
        type=int,
        default=0,
        help="Limit classifier to top-k prototypes per class (0 disables).",
    )
    parser.add_argument(
        "--prune-inactive-prototypes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prune inactive prototypes before visualization.",
    )
    parser.add_argument(
        "--prune-min-weight",
        type=float,
        default=0.0,
        help="Minimum positive classifier weight to keep a prototype when pruning.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Directory to save per-prototype grid images.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed prototype search using torch.distributed.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="World size (overridden by WORLD_SIZE env var).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for image sampling.",
    )
    parser.add_argument(
        "--use-raw-transforms",
        action="store_true",
        help="Use raw tensor transforms (no normalization).",
    )
    parser.add_argument(
        "--use-deit-transforms",
        action="store_true",
        help="Use DeiT evaluation transforms (resize 256 -> center-crop 224, etc.).",
    )
    parser.add_argument(
        "--save-bboxes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save input images with prototype bounding boxes and a CSV manifest.",
    )
    parser.add_argument(
        "--max-bbox-images",
        type=int,
        default=50,
        help="Maximum number of bbox images to save in total (0 disables).",
    )
    parser.add_argument(
        "--bboxes-per-image",
        type=int,
        default=3,
        help="Number of prototype bounding boxes to draw per saved image.",
    )
    parser.add_argument(
        "--save-composite",
        action="store_true",
        help="Save a composite image with bboxes and corresponding grids on the right.",
    )
    parser.add_argument(
        "--composite-link-lines",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw linking lines between bboxes and grid panels in composites.",
    )
    parser.add_argument(
        "--save-match-grid",
        action="store_true",
        help="Save a grid with the base image + bbox and nearest matches (replaces boxes output).",
    )
    parser.add_argument(
        "--match-grid-columns",
        type=int,
        default=4,
        help="Number of nearest-match columns to include per row in match grids.",
    )
    parser.add_argument(
        "--match-grid-padding",
        type=int,
        default=2,
        help="Padding (pixels) between cells in match grids.",
    )
    parser.add_argument(
        "--per-class-topk",
        action="store_true",
        help="Generate top-k prototypes per class instead of per-prototype grids.",
    )
    parser.add_argument(
        "--per-class-topk-k",
        type=int,
        default=5,
        help="Number of top prototypes per class to generate when --per-class-topk is set.",
    )
    parser.add_argument(
        "--per-class-grid-columns",
        type=int,
        default=0,
        help="Columns for the per-class grid (0 auto-squares).",
    )

    args = parser.parse_args()
    logger.info(f"Arguments: {args}")

    if not args.pipnet_checkpoint_path.is_file():
        logger.error("Checkpoint file not found at: %s", args.pipnet_checkpoint_path)
        return

    if args.distributed and args.world_size > 1:
        mp.spawn(
            fn=_distributed_worker,
            args=(args,),
            nprocs=args.world_size,
            join=True,
        )
    else:
        visualize_protoquant(args)


if __name__ == "__main__":
    main()
