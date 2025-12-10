from typing import Tuple
import torch
import torch.nn as nn

from src.models.proto_quantnet import ProtoQuantNet, ProtoQuantOutput
import logging

logger = logging.getLogger(__name__)


class TrainingWrapper(torch.nn.Module):
    """Wrapper to extract logits from ProtoQuantOutput for compatibility with training loops."""
    
    def __init__(self, model_to_wrap: ProtoQuantNet):
        super().__init__()
        self.model_to_wrap = model_to_wrap
        self.last_out: ProtoQuantOutput | None = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that returns only logits.
        
        Args:
            x: Input tensor
            
        Returns:
            Logits tensor for classification
        """
        out = self.model_to_wrap(x)
        self.last_out = out  # Store full output for later inspection if needed
        return out.logits


def _load_protoquant_from_checkpoint(
    backbone: nn.Module,
    checkpoint_path: str,
    num_classes: int,
    device: torch.device,
    train_codebook: bool = False,
    temperature: float = 0.1,
    classifier_sparsity_lambda: float = 0.0,
) -> ProtoQuantNet:
    """Load ProtoQuantNet model from a checkpoint.
    
    Args:
        backbone: Feature extraction network
        checkpoint_path: Path to checkpoint file
        num_classes: Number of output classes
        device: Device to load model onto
        train_codebook: Whether prototypes should be trainable
        temperature: Temperature for softmax pooling
        classifier_sparsity_lambda: L1 regularization strength on classifier weights
        
    Returns:
        Loaded ProtoQuantNet model
    """
    logger.info(f"Loading ProtoQuantNet model from: {checkpoint_path}")
    
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

    # Extract codebook
    if "codes" not in model_state_dict:
        raise ValueError("Loaded model state_dict is missing 'codes' key.")
    
    code_tensor = model_state_dict["codes"]

    # Build model
    model = ProtoQuantNet(
        backbone=backbone,
        codes=code_tensor.to(device),
        num_classes=num_classes,
        train_codebook=train_codebook,
        temperature=temperature,
        bias=False,
        classifier_sparsity_lambda=classifier_sparsity_lambda,
        freeze_backbone=True,
    ).to(device)
    
    # Load weights
    incompatible_keys = model.load_state_dict(model_state_dict, strict=False)
    logger.info(f"Loaded state_dict. Incompatible keys: {incompatible_keys}")

    return model


def _build_protoquant_from_codebook(
    backbone: nn.Module,
    codebook_path: str,
    num_classes: int,
    device: torch.device,
    train_codebook: bool = False,
    temperature: float = 0.1,
    classifier_sparsity_lambda: float = 0.0,
) -> ProtoQuantNet:
    """Build ProtoQuantNet from backbone and codebook file.
    
    Args:
        backbone: Feature extraction network
        codebook_path: Path to codebook tensor file
        num_classes: Number of output classes
        device: Device to load model onto
        train_codebook: Whether prototypes should be trainable
        temperature: Temperature for softmax pooling
        classifier_sparsity_lambda: L1 regularization strength on classifier weights
        
    Returns:
        New ProtoQuantNet model
    """
    logger.info(f"Building ProtoQuantNet from backbone and codebook: {codebook_path}")
    
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

    model = ProtoQuantNet(
        backbone=backbone,
        codes=code_tensor.to(device),
        num_classes=num_classes,
        train_codebook=train_codebook,
        temperature=temperature,
        bias=False,
        classifier_sparsity_lambda=classifier_sparsity_lambda,
        freeze_backbone=True,
    ).to(device)
    
    return model


def build_pipnet_model(
    backbone: nn.Module,
    num_classes: int,
    device: torch.device,
    pipnet_checkpoint_path: str | None = None,
    codebook_path: str | None = None,
    train_codebook: bool = False,
    temperature: float = 0.1,
    classifier_sparsity_lambda: float = 0.0,
    use_random_codes: bool = False,
) -> ProtoQuantNet:
    """Build ProtoQuantNet model from checkpoint or codebook.

    Args:
        backbone: Feature extraction network (e.g., ResNet without fc layer)
        num_classes: Number of output classes
        device: Device to load model onto
        pipnet_checkpoint_path: Path to full model checkpoint (optional)
        codebook_path: Path to codebook weights only (optional)
        train_codebook: Whether prototypes should be trainable
        temperature: Temperature for softmax pooling (default: 0.1)
        classifier_sparsity_lambda: L1 regularization on classifier weights (default: 0.0)
        use_random_codes: Replace loaded codes with random normal distribution (default: False)

    Returns:
        ProtoQuantNet model
        
    Raises:
        ValueError: If neither checkpoint_path nor codebook_path is provided
    """

    # Freeze backbone (standard procedure for prototype networks)
    for p in backbone.parameters():
        p.requires_grad = False

    # Build model from checkpoint or codebook
    if pipnet_checkpoint_path:
        model = _load_protoquant_from_checkpoint(
            backbone=backbone,
            checkpoint_path=pipnet_checkpoint_path,
            num_classes=num_classes,
            device=device,
            train_codebook=train_codebook,
            temperature=temperature,
            classifier_sparsity_lambda=classifier_sparsity_lambda,
        )
    elif codebook_path:
        model = _build_protoquant_from_codebook(
            backbone=backbone,
            codebook_path=codebook_path,
            num_classes=num_classes,
            device=device,
            train_codebook=train_codebook,
            temperature=temperature,
            classifier_sparsity_lambda=classifier_sparsity_lambda,
        )
    else:
        raise ValueError(
            "Must provide either 'pipnet_checkpoint_path' or 'codebook_path'."
        )
    
    # Replace codes with random normal distribution if requested
    if use_random_codes:
        logger.info(f"Replacing {model.num_prototypes} codes with random normal distribution")
        random_codes = torch.randn_like(model.codes)
        if isinstance(model.codes, nn.Parameter):
            model.codes = nn.Parameter(random_codes, requires_grad=model.codes.requires_grad)
        else:
            model.register_buffer("codes", random_codes)
        logger.info("Codes replaced with random values")

    return model
