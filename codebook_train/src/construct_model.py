from src.models.adapters import guess_adapter
from src.config.main_config import MainConfig
import torch
from torch import nn
from src.models_registry import MODELS
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _load_checkpoint(checkpoint_path: str, device: torch.device) -> object:
    """Load a checkpoint from disk.

    Tries to use `weights_only=True` (PyTorch 2.0+) for safety/performance, and
    falls back to a normal `torch.load` if that argument is unsupported.

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Device (or map_location) to load tensors onto.

    Returns:
        The deserialized checkpoint object. Commonly either:
        - a raw state_dict: `dict[str, Tensor]`, or
        - a training bundle dict containing a state_dict under keys like
          `state_dict`, `model_state_dict`, or `model`.
    """
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    try:
        ckpt = torch.load(checkpoint_path, weights_only=True, map_location=device)
        logger.info(f"Type of checkpoint loaded with weights_only=True: {type(ckpt)}")
        return ckpt
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def _extract_state_dict(ckpt: object) -> dict[str, torch.Tensor]:
    """Extract a tensor state_dict from a checkpoint object.

    Supports common formats:
    - Raw `state_dict` saved directly.
    - Training bundles with weights under `state_dict`, `model_state_dict`, or `model`.
    - Bundles where `model` is an `nn.Module`-like object (uses `.state_dict()`).

    Also strips common DistributedDataParallel prefixes (`module.`, `model.`) and
    drops any non-tensor entries.

    Args:
        ckpt: The loaded checkpoint object.

    Returns:
        A cleaned mapping `dict[str, Tensor]` suitable for `nn.Module.load_state_dict`.

    Raises:
        ValueError: If a tensor mapping cannot be found in the checkpoint.
    """
    if isinstance(ckpt, dict):
        logger.info(f"Key(s) in the loaded checkpoint: {list(ckpt.keys())}")
    else:
        logger.info("Key(s) in the loaded checkpoint: N/A")

    # Handle common checkpoint layouts.
    # IMPORTANT: Prefer an actual tensor mapping (usually under 'state_dict').
    state_dict_obj: object = ckpt
    chosen_layout = "raw"

    if isinstance(ckpt, dict):
        # 1) If the checkpoint dict itself looks like a state_dict, use it.
        if ckpt and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state_dict_obj = ckpt
            chosen_layout = "ckpt"
        else:
            # 2) Common keys (prefer 'state_dict' over 'model')
            candidates: list[tuple[str, Any]] = []
            for key in ("state_dict", "model_state_dict", "model"):
                if key in ckpt:
                    candidates.append((key, ckpt[key]))

            # 3) Some checkpoints store a module under 'model'
            for name, candidate in candidates:
                if isinstance(candidate, dict):
                    state_dict_obj = candidate
                    chosen_layout = name
                    break
                state_dict_fn = getattr(candidate, "state_dict", None)
                if callable(state_dict_fn):
                    state_dict_obj = state_dict_fn()
                    chosen_layout = f"{name}.state_dict()"
                    break

            # 4) Nested dict layouts (rare but seen)
            if not isinstance(state_dict_obj, dict) and isinstance(ckpt.get("model"), dict):
                nested = ckpt["model"]
                if "state_dict" in nested and isinstance(nested["state_dict"], dict):
                    state_dict_obj = nested["state_dict"]
                    chosen_layout = "model.state_dict"

    logger.info(f"Selected checkpoint layout: {chosen_layout}")
    logger.info(f"type of state_dict to be loaded: {type(state_dict_obj)}")
    logger.info(
        f"Keys in the state_dict to be loaded: {list(state_dict_obj.keys()) if isinstance(state_dict_obj, dict) else 'N/A'}"
    )

    if not isinstance(state_dict_obj, dict):
        raise ValueError("Checkpoint did not contain a state_dict")

    # Keep only tensor entries; strip common DataParallel/Distributed prefixes
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict_obj.items():
        if not isinstance(value, torch.Tensor):
            continue
        cleaned_key = key
        for prefix in ("module.", "model."):
            if cleaned_key.startswith(prefix):
                cleaned_key = cleaned_key[len(prefix):]
        cleaned[cleaned_key] = value

    return cleaned


def _remap_vit_norm_keys(state_dict: dict[str, torch.Tensor], model: nn.Module) -> dict[str, torch.Tensor]:
    """Remap ViT normalization layer keys between 'norm' and 'fc_norm'.

    When timm ViT models use global_pool='avg', the final layer norm is named
    'fc_norm' instead of 'norm'. This function handles the bidirectional remapping
    so checkpoints saved with one convention can be loaded into models using the other.

    Args:
        state_dict: The state dict to potentially remap.
        model: The target model to check for expected key names.

    Returns:
        A state dict with remapped keys if necessary.
    """
    model_keys = set(model.state_dict().keys())
    remapped = {}

    # Define the possible remappings (bidirectional)
    remap_rules = {
        "norm.weight": "fc_norm.weight",
        "norm.bias": "fc_norm.bias",
        "fc_norm.weight": "norm.weight",
        "fc_norm.bias": "norm.bias",
    }

    for key, value in state_dict.items():
        new_key = key
        if key in remap_rules:
            candidate = remap_rules[key]
            # Only remap if the candidate exists in model but original doesn't
            if candidate in model_keys and key not in model_keys:
                new_key = candidate
                logger.info(f"Remapping state_dict key: '{key}' -> '{new_key}'")
        remapped[new_key] = value

    return remapped


def register_models() -> None:
    """Register all the models in the registry by importing the module"""
    import src.models  # noqa: F401


def construct_model(cfg: MainConfig, device: torch.device) -> nn.Module:
    """Construct the model based on the configuration.

    Args:
        cfg (MainConfig):  The main configuration object containing model settings.
        device (torch.device): The device to run the model on.

    Returns:
        nn.Module: A PyTorch model initialized with the specified architecture and loaded with weights from a checkpoint.
    """

    return construct_model_no_cfg(
        model_name=cfg.model.name,
        num_classes=cfg.dataset.num_classes,
        device=device,
        checkpoint_path=cfg.model.checkpoint_path,
        global_pool=cfg.model.global_pool,
    )

def construct_model_no_cfg(model_name: str, num_classes: int, device: torch.device, checkpoint_path: str | None = None, global_pool: str = "") -> nn.Module:
    """Construct the model based on the provided parameters.

    Args:
        model_name (str): The name of the model architecture.
        num_classes (int): The number of output classes for the model.
        device (torch.device): The device to run the model on.
        checkpoint_path (str | None): Optional path to a checkpoint to load weights from.
        global_pool (str): Optional global pooling method to use.

    Returns:
        nn.Module: A PyTorch model initialized with the specified architecture and loaded with weights from a checkpoint if provided.
    """

    model: nn.Module = MODELS[model_name](num_classes=num_classes)
    if global_pool != "":
        reset_classifier = getattr(model, "reset_classifier", None)
        if callable(reset_classifier):
            logger.info(f"Resetting classifier with global_pool={global_pool}")
            reset_classifier(num_classes=num_classes, global_pool=global_pool)
        else:
            raise ValueError(
                f"Model {model_name} does not support reset_classifier(); cannot apply global_pool={global_pool}."
            )

    model.to(device)

    if checkpoint_path:
        ckpt = _load_checkpoint(checkpoint_path, device)
        try:
            cleaned = _extract_state_dict(ckpt)
        except ValueError as e:
            raise ValueError(
                f"Unsupported checkpoint format at {checkpoint_path}: expected a state_dict or dict containing one"
            ) from e

        # Remap ViT norm keys if needed (handles norm <-> fc_norm mismatch)
        cleaned = _remap_vit_norm_keys(cleaned, model)

        model.load_state_dict(cleaned, strict=True)
    else:
        logger.info("No checkpoint_path provided. The model will use its default initialization.")

    return model

def get_backbone(model: nn.Module) -> nn.Module:
    """Extracts the backbone (feature extractor) from a given model.

    Args:
        model (nn.Module): The full model from which to extract the backbone.

    Returns:
        nn.Module: The backbone of the model.
    """
    
    adapter = guess_adapter(model)
    backbone = adapter.extract_features(model)
    logger.info(
        f"Using adapter {adapter.__class__.__name__} for model {model.__class__.__name__} to extract backbone."
    )
    
    return backbone