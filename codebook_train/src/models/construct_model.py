from src.models.inaturalist_resnet import inaturalist_resnet50
from src.config.main_config import MainConfig
import torch
from torch import nn
from torchvision import models
from src.models.convnextv2_backbone import (
    convnextv2_tiny_backbone,
    convnextv2_nano_backbone,
)

import logging

logger = logging.getLogger(__name__)

MODELS = {
    "convnext_tiny": models.convnext_tiny,
    "convnext_large": models.convnext_large,
    "convnextv2_tiny_backbone": convnextv2_tiny_backbone,
    "convnextv2_nano_backbone": convnextv2_nano_backbone,
    "resnet50": models.resnet50,
    "inaturalist_resnet50": inaturalist_resnet50
}


def construct_model(cfg: MainConfig, device: torch.device) -> nn.Module:
    """Construct the model based on the configuration.

    Args:
        cfg (MainConfig):  The main configuration object containing model settings.
        device (torch.device): The device to run the model on.

    Returns:
        nn.Module: A PyTorch model initialized with the specified architecture and loaded with weights from a checkpoint.
    """

    model: nn.Module = MODELS[cfg.model.name](num_classes=cfg.dataset.num_classes)
    model.to(device)

    logger.info(f"Loading model from checkpoint: {cfg.model.checkpoint_path}")
    state_dict = torch.load(
        cfg.model.checkpoint_path,
        weights_only=True,
        map_location=device,
    )

    if "model" in state_dict:
        state_dict = state_dict["model"]

    model.load_state_dict(state_dict, strict=True)

    return model
