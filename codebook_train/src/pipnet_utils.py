from typing import Tuple
import torch
import torch.nn as nn

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
        self._last_proto_fvec = None
    
    # This forward pass calls the underlying model and returns only the logits
    def forward(self, x):
        out = self.model_to_wrap(x)
        # Cache proto vector for auxiliary regularization during training
        self._last_proto_fvec = out.proto_fvec
        return out.logits


def _load_pipnet_from_checkpoint(
    backbone: nn.Module,
    checkpoint_path: str,
    num_classes: int,
    device: torch.device,
    train_codebook: bool = False
) -> Tuple[PIPNetWrapper, nn.Module]:
    """
    Loads a full PIPNet model from a .pth checkpoint.
    """
    logger.info(f"Loading pre-trained PIPNet model from: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Handle cases where the state_dict is nested under 'model' or 'state_dict'
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            model_state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            model_state_dict = ckpt["state_dict"]
        else:
            model_state_dict = ckpt
    else:
        # Fallback if it's not a dict (unlikely for state_dicts)
        model_state_dict = ckpt

    # Strip DataParallel prefixes (module. or model_to_wrap.)
    prefix_to_strip = ["module.", "model_to_wrap."]
    for p in prefix_to_strip:
        if any(k.startswith(p) for k in model_state_dict.keys()):
            logger.info(f"Stripping '{p}' prefix from state_dict keys.")
            model_state_dict = {
                k[len(p):]: v 
                for k, v in model_state_dict.items() 
                if k.startswith(p)
            }

    # Extract codebook to reconstruct the head
    # Note: We look for "head.codebook" because that's where it lives in the wrapper
    if "head.codebook" not in model_state_dict:
        raise ValueError("Loaded model state_dict is missing 'head.codebook'.")
    
    code_tensor = model_state_dict["head.codebook"]

    # Re-instantiate the head
    # We use the explicitly passed 'train_codebook' arg, allowing override of saved state
    pip_head = QuantizedPIPNetHead(
        num_classes=num_classes,
        codebook=code_tensor.to(device),
        normalize=True,
        bias=False,
        train_codebook=train_codebook, 
    ).to(device)

    model = PIPNetWrapper(backbone=backbone, head=pip_head).to(device)
    
    # Load weights
    incompatible_keys = model.load_state_dict(model_state_dict, strict=False)
    logger.info(f"Loaded state_dict. Incompatible keys: {incompatible_keys}")

    return model, pip_head


def _build_pipnet_from_codebook(
    backbone: nn.Module,
    codebook_path: str,
    num_classes: int,
    device: torch.device,
    train_codebook: bool = False
) -> Tuple[PIPNetWrapper, nn.Module]:
    """
    Builds a new PIPNet using a backbone and a standalone codebook tensor file.
    """
    logger.info(f"Building new PIPNet model from backbone and codebook: {codebook_path}")
    
    ckpt = torch.load(codebook_path, map_location=device)
    
    # Handle different ways the codebook might be saved
    if isinstance(ckpt, dict) and "embeddings.weight" in ckpt:
        code_tensor = ckpt["embeddings.weight"]
    elif isinstance(ckpt, torch.Tensor):
        code_tensor = ckpt
    else:
        raise ValueError(f"Could not find code tensor in {codebook_path}")

    if code_tensor.ndim != 2:
        raise ValueError(f"Unexpected code tensor shape: {tuple(code_tensor.shape)}")

    pip_head = QuantizedPIPNetHead(
        num_classes=num_classes,
        codebook=code_tensor.to(device),
        normalize=True,
        bias=False,
        train_codebook=train_codebook,
    ).to(device)

    model = PIPNetWrapper(backbone=backbone, head=pip_head).to(device)
    return model, pip_head


def build_pipnet_model(
    backbone: nn.Module,
    num_classes: int,
    device: torch.device,
    pipnet_checkpoint_path: str | None = None,
    codebook_path: str | None = None,
    train_codebook: bool = False
) -> Tuple[PIPNetWrapper, nn.Module]:
    """
    Main entry point to build/load the PIPNet model.

    Args:
        backbone (nn.Module): The feature extractor (e.g., ResNet without fc layer).
                              Must imply strict=False when loading weights if provided.
        num_classes (int): Number of output classes.
        device (torch.device): Device to load tensors onto.
        pipnet_checkpoint_path (str, optional): Path to full .pth model checkpoint.
        codebook_path (str, optional): Path to just the codebook weights.
        train_codebook (bool): Whether the prototypes should be trainable.

    Returns:
        model (PIPNetWrapper): The full model.
        pip_head (QuantizedPIPNetHead): The head component.
    """

    # 1. Freeze backbone (standard PIPNet procedure)
    for p in backbone.parameters():
        p.requires_grad = False

    # 2. Dispatch
    if pipnet_checkpoint_path:
        model, pip_head = _load_pipnet_from_checkpoint(
            backbone=backbone,
            checkpoint_path=pipnet_checkpoint_path,
            num_classes=num_classes,
            device=device,
            train_codebook=train_codebook
        )
    elif codebook_path:
        model, pip_head = _build_pipnet_from_codebook(
            backbone=backbone,
            codebook_path=codebook_path,
            num_classes=num_classes,
            device=device,
            train_codebook=train_codebook
        )
    else:
        raise ValueError(
            "Must provide either 'pipnet_checkpoint_path' or 'codebook_path'."
        )

    return model, pip_head
