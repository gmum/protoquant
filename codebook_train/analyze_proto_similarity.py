# analyze_proto_similarity.py

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from src.construct_model import construct_model_no_cfg, get_backbone
from src.datasets.construct_dataset import get_dataset
from src.datasets.transforms import get_transforms_by_mode
from src.pipnet_utils import build_pipnet_model
from other_benchmarks.purity_benchmark.pipnet.pipnet import PIPNet, get_network

logger = logging.getLogger(__name__)


def _ensure_list_batches(val_loader: Iterable) -> Iterable:
    return val_loader


def _summarize_similarity(sim_map: torch.Tensor, top_k: int) -> torch.Tensor:
    """Summarize similarity map per sample using top-k prototype maxes.

    Args:
        sim_map: (B, P, H, W)
        top_k: number of prototypes to average after spatial max
    """
    if sim_map.ndim != 4:
        raise ValueError(f"Expected 4D similarity map, got {sim_map.ndim}D")

    proto_max = sim_map.amax(dim=(-1, -2))  # (B, P)
    k = min(top_k, proto_max.shape[1])
    if k < 1:
        k = proto_max.shape[1]
    top_vals, _ = torch.topk(proto_max, k=k, dim=1)
    return top_vals.mean(dim=1)  # (B,)


def _cosine_similarity_map(
    features: torch.Tensor, prototypes: torch.Tensor
) -> torch.Tensor:
    """Compute cosine similarity map between feature vectors and prototype vectors.

    Args:
        features: (B, C, H, W)
        prototypes: (P, C)
    """
    if features.ndim != 4:
        raise ValueError(f"Expected 4D features, got {features.ndim}D")

    eps = 1e-6
    b, c, h, w = features.shape
    x = features.permute(0, 2, 3, 1).reshape(-1, c)
    x_unit = x / (x.norm(dim=-1, keepdim=True) + eps)
    p_unit = prototypes / (prototypes.norm(dim=-1, keepdim=True) + eps)
    sim = x_unit @ p_unit.t()  # (B*H*W, P)
    return sim.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()


def _pipnet_cosine_map(model: PIPNet, x: torch.Tensor) -> torch.Tensor:
    """Approximate cosine similarity map for benchmark PIP-Net.

    If a 1x1 conv exists in add_on, use its weights as prototype vectors.
    Otherwise, use identity basis prototypes (channels), yielding normalized
    channel activations as cosine similarities.
    """
    features = model._net(x)
    if features.ndim != 4:
        raise ValueError(f"Expected CNN features for PIP-Net, got {features.ndim}D")

    conv_layer = None
    if isinstance(model._add_on, nn.Sequential) and len(model._add_on) > 0:
        if isinstance(model._add_on[0], nn.Conv2d):
            conv_layer = model._add_on[0]

    if conv_layer is None:
        eps = 1e-6
        return features / (features.norm(dim=1, keepdim=True) + eps)

    weight = conv_layer.weight.squeeze(-1).squeeze(-1)  # (P, C)
    return _cosine_similarity_map(features, weight)


def _load_protoquant(
    *,
    checkpoint_path: str,
    backbone_name: str,
    num_classes: int,
    device: torch.device,
    global_pool: str,
    temperature: float,
) -> nn.Module:
    base_model = construct_model_no_cfg(
        model_name=backbone_name,
        num_classes=num_classes,
        device=device,
        checkpoint_path=None,
        global_pool=global_pool,
    )
    backbone = get_backbone(base_model)
    model = build_pipnet_model(
        backbone=backbone,
        num_classes=num_classes,
        device=device,
        pipnet_checkpoint_path=checkpoint_path,
        codebook_path=None,
        train_codebook=False,
        temperature=temperature,
    )
    model.eval()
    return model


def _strip_module_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _load_pipnet(
    *,
    checkpoint_path: str,
    net_name: str,
    num_classes: int,
    num_features: int,
    bias: bool,
    disable_pretrained: bool,
    device: torch.device,
) -> PIPNet:
    args = SimpleNamespace(
        net=net_name,
        num_features=num_features,
        bias=bias,
        disable_pretrained=disable_pretrained,
    )
    features, add_on_layers, pool_layer, classification_layer, num_prototypes = (
        get_network(num_classes, args)
    )
    model = PIPNet(
        num_classes=num_classes,
        num_prototypes=num_prototypes,
        feature_net=features,
        args=args,
        add_on_layers=add_on_layers,
        pool_layer=pool_layer,
        classification_layer=classification_layer,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    state_dict = _strip_module_prefix(state_dict)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        logger.warning(
            "PIP-Net state_dict load with missing keys: %s, unexpected keys: %s",
            incompatible.missing_keys,
            incompatible.unexpected_keys,
        )

    model.to(device)
    model.eval()
    return model


def analyze_similarity(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading ProtoQuant model...")
    protoquant = _load_protoquant(
        checkpoint_path=args.protoquant_checkpoint_path,
        backbone_name=args.protoquant_backbone,
        num_classes=args.num_classes,
        device=device,
        global_pool=args.protoquant_global_pool,
        temperature=args.protoquant_temperature,
    )

    logger.info("Loading PIP-Net model...")
    pipnet = _load_pipnet(
        checkpoint_path=args.pipnet_checkpoint_path,
        net_name=args.pipnet_net,
        num_classes=args.num_classes,
        num_features=args.pipnet_num_features,
        bias=args.pipnet_bias,
        disable_pretrained=args.pipnet_disable_pretrained,
        device=device,
    )

    logger.info("Preparing validation dataset...")
    is_cub = args.dataset_name.lower() == "cub200"
    transform_mode = (
        "deit"
        if args.use_deit_transforms
        else "raw"
        if args.use_raw_transforms
        else args.transforms_mode
    )

    _, val_transform = get_transforms_by_mode(
        transform_mode,
        model_name=args.transforms_model_name or args.protoquant_backbone,
        resize_size=224 if is_cub else 256,
        crop_size=None if is_cub else 224,
        random_erase=None,
        horizontal_flip=None,
        is_precropped=is_cub,
        autoaugment=False,
    )

    _, val_ds = get_dataset(
        name=args.dataset_name,
        path=args.dataset_path,
        train_transform=val_transform,
        val_transform=val_transform,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    protoquant_scores: list[float] = []
    pipnet_scores: list[float] = []

    logger.info("Computing cosine similarity summaries...")
    max_batches = (
        args.max_batches if args.max_batches and args.max_batches > 0 else None
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(_ensure_list_batches(val_loader))):
            images = batch[0].to(device)

            proto_out = protoquant(images, return_similarity_map=True)
            if proto_out.similarity_map is None:
                raise RuntimeError("ProtoQuant did not return similarity_map")
            pq_scores = _summarize_similarity(proto_out.similarity_map, args.top_k)

            pip_sim_map = _pipnet_cosine_map(pipnet, images)
            pp_scores = _summarize_similarity(pip_sim_map, args.top_k)

            protoquant_scores.extend(pq_scores.detach().cpu().tolist())
            pipnet_scores.extend(pp_scores.detach().cpu().tolist())

            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break

    scores_path = output_dir / "proto_similarity_scores.npz"
    np.savez(
        scores_path,
        protoquant=np.array(protoquant_scores, dtype=np.float32),
        pipnet=np.array(pipnet_scores, dtype=np.float32),
    )
    logger.info("Saved similarity scores to %s", scores_path)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.boxplot(
        [protoquant_scores, pipnet_scores],
        labels=["ProtoQuant (cosine)", "PIP-Net (proxy cosine)"],
        showmeans=True,
    )
    ax.set_ylabel(f"Top-{args.top_k} mean cosine similarity")
    ax.set_title("Prototype-to-input cosine similarity comparison")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    plot_path = output_dir / "proto_similarity_boxplot.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    logger.info("Saved boxplot to %s", plot_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare ProtoQuant vs PIP-Net prototype similarity via cosine maps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--protoquant-checkpoint-path",
        type=str,
        required=True,
        help="Path to ProtoQuant checkpoint (.pth) with codes.",
    )
    parser.add_argument(
        "--protoquant-backbone",
        type=str,
        required=True,
        help="Backbone architecture name for ProtoQuant (model registry name).",
    )
    parser.add_argument(
        "--protoquant-global-pool",
        type=str,
        default="",
        help="Optional global_pool for timm ViT backbones.",
    )
    parser.add_argument(
        "--protoquant-temperature",
        type=float,
        default=0.1,
        help="ProtoQuant temperature (used for pooling in forward).",
    )

    parser.add_argument(
        "--pipnet-checkpoint-path",
        type=str,
        required=True,
        help="Path to PIP-Net benchmark checkpoint (.pth).",
    )
    parser.add_argument(
        "--pipnet-net",
        type=str,
        default="convnext_tiny_13",
        help="PIP-Net backbone name (see purity_benchmark/pipnet/pipnet.py).",
    )
    parser.add_argument(
        "--pipnet-num-features",
        type=int,
        default=0,
        help="PIP-Net num_features (0 for original).",
    )
    parser.add_argument(
        "--pipnet-bias",
        action="store_true",
        help="Enable bias in PIP-Net classification layer.",
    )
    parser.add_argument(
        "--pipnet-disable-pretrained",
        action="store_true",
        help="Disable pretrained weights in PIP-Net backbone.",
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="cub200",
        help="Dataset name (e.g., cub200).",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Root dataset path for validation split.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        required=True,
        help="Number of classes in the dataset.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument(
        "--transforms-mode",
        type=str,
        default="default",
        choices=["default", "deit", "raw", "resize_norm"],
        help="Transform pipeline for evaluation.",
    )
    parser.add_argument(
        "--use-deit-transforms",
        action="store_true",
        help="Use DeiT eval transforms (overrides --transforms-mode).",
    )
    parser.add_argument(
        "--use-raw-transforms",
        action="store_true",
        help="Use raw tensor transforms (overrides --transforms-mode).",
    )
    parser.add_argument(
        "--transforms-model-name",
        type=str,
        default=None,
        help="Model name used by transform selection (defaults to protoquant backbone).",
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Top-k prototypes to average."
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional limit on number of batches for quick runs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_results",
        help="Output directory for plots and scores.",
    )

    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = build_parser()
    args = parser.parse_args()

    analyze_similarity(args)


if __name__ == "__main__":
    main()
