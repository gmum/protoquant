from src.models.adapters import guess_adapter
from src.config.main_config import MainConfig
import torch
from torch import nn
from src.models_registry import MODELS
import logging

logger = logging.getLogger(__name__)


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
        logger.info(f"Resetting classifier with global_pool={global_pool}")
        model.reset_classifier(num_classes=num_classes, global_pool=global_pool)

    model.to(device)

    if checkpoint_path:
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        state_dict = torch.load(
            checkpoint_path,
            weights_only=True,
            map_location=device,
        )

        # Handle checkpoints that save the model under a 'model' key
        if "model" in state_dict:
            state_dict = state_dict["model"]

        model.load_state_dict(state_dict, strict=True)
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