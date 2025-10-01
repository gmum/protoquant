from typing import Tuple
import torch
import torch.nn as nn

from src.models.adapters import guess_adapter
from src.config.pipnet_config import PipNetConfig
from src.construct_model import construct_model
from src.models.pipnet import QuantizedPIPNetHead
import logging

logger = logging.getLogger(__name__)


class PIPNetWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, head: QuantizedPIPNetHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        out_obj = self.head(feats)
        return out_obj

    def train(self, mode: bool = True):
        # Keep head in requested mode, but freeze backbone stats
        super().train(mode)
        self.backbone.eval()
        return self
    

class TrainingWrapper(torch.nn.Module):
    def __init__(self, model_to_wrap):
        super().__init__()
        self.model_to_wrap = model_to_wrap
    
    # This forward pass calls the underlying model and returns only the logits
    def forward(self, x):
        return self.model_to_wrap(x).logits


def _load_pipnet_from_checkpoint(
    cfg: PipNetConfig, backbone: nn.Module, device: torch.device
) -> Tuple[PIPNetWrapper, QuantizedPIPNetHead]:
    """
    Loads a full PIPNet model from a checkpoint, re-instantiating the head
    to respect the current `train_codebook` configuration.
    """
    if not cfg.pipnet_checkpoint_path:
        raise ValueError("cfg.pipnet_checkpoint_path must be set to load from checkpoint.")
        
    logger.info(
        f"Loading pre-trained PIPNet model from: {cfg.pipnet_checkpoint_path}"
    )
    ckpt = torch.load(cfg.pipnet_checkpoint_path, map_location=device)
    if "model" not in ckpt:
        raise ValueError("Checkpoint is missing the 'model' state_dict key.")
    model_state_dict = ckpt["model"]

    # Extract codebook from the loaded state_dict to reconstruct the head
    if "head.codebook" not in model_state_dict:
        raise ValueError("Loaded model state_dict is missing 'head.codebook'.")
    code_tensor = model_state_dict["head.codebook"]

    # Build the head using the loaded codebook tensor, but crucially, use the
    # train_codebook setting from the *current* config. This allows overriding.
    pip_head = QuantizedPIPNetHead(
        num_classes=cfg.dataset.num_classes,
        codebook=code_tensor.to(device),
        normalize=True,
        bias=False,
        train_codebook=cfg.training.train_codebook,  # <-- OVERRIDE HAPPENS HERE
    ).to(device)

    model = PIPNetWrapper(backbone=backbone, head=pip_head).to(device)
    incompatible_keys = model.load_state_dict(model_state_dict, strict=False)
    logger.info(
        f"Successfully loaded state_dict. "
        f"Incompatible keys (expected if train_codebook changed): {incompatible_keys}"
    )

    return model, pip_head  


def _build_pipnet_from_codebook(
    cfg: PipNetConfig, backbone: nn.Module, device: torch.device
) -> Tuple[PIPNetWrapper, QuantizedPIPNetHead]:
    """
    Builds a new PIPNet model from a backbone and a separate codebook tensor file.
    """
    if not cfg.codebook_path:
        raise ValueError("cfg.codebook_path must be set to build from a codebook.")

    logger.info("Building new PIPNet model from backbone and codebook file.")
    ckpt = torch.load(cfg.codebook_path, map_location=device, weights_only=True)
    if "embeddings.weight" not in ckpt:
        raise ValueError(
            f"Codebook checkpoint missing 'embeddings.weight' key: {list(ckpt.keys())[:5]}..."
        )

    code_tensor: torch.Tensor = ckpt["embeddings.weight"]
    if code_tensor.ndim != 2:
        raise ValueError(f"Unexpected code tensor shape: {tuple(code_tensor.shape)}")
    logger.info(
        f"Loaded codebook tensor from {cfg.codebook_path} with shape {tuple(code_tensor.shape)}"
    )

    # Build PIPNet head
    pip_head = QuantizedPIPNetHead(
        num_classes=cfg.dataset.num_classes,
        codebook=code_tensor.to(device),  # register as buffer inside head
        normalize=True,
        bias=False,
        train_codebook=cfg.training.train_codebook,
    ).to(device)

    model = PIPNetWrapper(backbone=backbone, head=pip_head).to(device)
    return model, pip_head


def build_pipnet_model(
    cfg: PipNetConfig, device: torch.device
) -> Tuple[PIPNetWrapper, QuantizedPIPNetHead]:
    """
    Builds or loads a PIPNet model based on the provided configuration.

    This function dispatches to helper functions based on whether a full PIPNet
    checkpoint is provided or if the model should be constructed from a backbone
    and a separate codebook.
    """


    # Validate configuration and handle redundancy
    if cfg.pipnet_checkpoint_path:
        if cfg.model.checkpoint_path:
            logger.warning(
                f"Both `pipnet_checkpoint_path` and `model.checkpoint_path` were provided. "
                f"The backbone's weights will be loaded from the full PIP-Net checkpoint, "
                f"so `model.checkpoint_path` ('{cfg.model.checkpoint_path}') will be IGNORED."
            )
        # We don't need to load initial backbone weights, as they will be overwritten.
        # Ensure construct_model uses no pretrained weights in this case.
        cfg.model.checkpoint_path = None

    # 1. Prepare the backbone architecture (weights will be loaded later if using pipnet_checkpoint_path)
    base_model = construct_model(cfg, device)  # type: ignore
    adapter = guess_adapter(base_model)
    logger.info(
        f"Using adapter {adapter.__class__.__name__} for model {cfg.model.name}"
    )
    backbone = adapter.extract_features(base_model)

    # Freeze backbone (train head and optionally codebook only)
    for p in backbone.parameters():
        p.requires_grad = False

    # 2. Dispatch to the appropriate construction method
    if cfg.pipnet_checkpoint_path:
        model, pip_head = _load_pipnet_from_checkpoint(cfg, backbone, device)
    elif cfg.codebook_path:
        model, pip_head = _build_pipnet_from_codebook(cfg, backbone, device)
    else:
        raise ValueError(
            "Configuration error: Must provide either 'pipnet_checkpoint_path' "
            "(to load a full model) or 'codebook_path' (to build a new one)."
        )

    return model, pip_head
