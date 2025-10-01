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

    model: nn.Module = MODELS[cfg.model.name](num_classes=cfg.dataset.num_classes)
    if cfg.model.global_pool != "":
        logger.info(f"Resetting classifier with global_pool={cfg.model.global_pool}")
        model.reset_classifier(num_classes=cfg.dataset.num_classes, global_pool=cfg.model.global_pool)

    model.to(device)

    if cfg.model.checkpoint_path:
        logger.info(f"Loading model from checkpoint: {cfg.model.checkpoint_path}")
        state_dict = torch.load(
            cfg.model.checkpoint_path,
            weights_only=True,
            map_location=device,
        )

        # Handle checkpoints that save the model under a 'model' key
        if "model" in state_dict:
            state_dict = state_dict["model"]

        model.load_state_dict(state_dict, strict=True)
    else:
        logger.info("No model.checkpoint_path provided. The model will use its default initialization.")

    return model
