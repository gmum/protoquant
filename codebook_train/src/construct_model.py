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
    "convnextv2_tiny_backbone": convnextv2_tiny_backbone,
    "convnextv2_nano_backbone": convnextv2_nano_backbone,
}


def construct_model(cfg: MainConfig) -> nn.Module:
    model: nn.Module = MODELS[cfg.model.name](num_classes=cfg.dataset.num_classes)

    logger.info(f"Loading model from checkpoint: {cfg.model.checkpoint_path}")
    state_dict = torch.load(cfg.model.checkpoint_path, weights_only=True)

    if "model" in state_dict:
        state_dict = state_dict["model"]

    model.load_state_dict(state_dict)

    return model
