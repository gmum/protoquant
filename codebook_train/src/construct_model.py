from src.config.main_config import MainConfig
import torch
from torch import nn
from torchvision import models

import logging

logger = logging.getLogger(__name__)

MODELS = {"convnext_tiny": models.convnext_tiny}


def construct_model(cfg: MainConfig) -> nn.Module:
    model: nn.Module = MODELS[cfg.model.name](num_classes=cfg.dataset.num_classes)

    logger.info(f"Loading model from checkpoint: {cfg.model.checkpoint_path}")
    model.load_state_dict(torch.load(cfg.model.checkpoint_path))

    logger.info(f"Loaded model: {model}")

    return model
