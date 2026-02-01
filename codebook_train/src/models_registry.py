from torchvision import models
import torch.nn as nn
from typing import Callable, TypeVar

# Create a TypeVar to preserve the specific signature of the decorated function
T = TypeVar("T", bound=Callable[..., nn.Module])

MODELS: dict[str, Callable[..., nn.Module]] = {}


def register_model(func: T) -> T:
    """Decorator to register a model in the MODELS dictionary."""
    MODELS[func.__name__] = func
    return func


@register_model
def convnext_tiny(**kwargs):
    return models.convnext_tiny(**kwargs)


@register_model
def convnext_large(**kwargs):
    return models.convnext_large(**kwargs)


@register_model
def resnet50(**kwargs):
    return models.resnet50(**kwargs)


@register_model
def vit_b_16(**kwargs):
    # GAP version built on top of torchvision ViT-B/16
    from src.models.gap_vit_b_16 import vit_b_16 as build_gap_vit_b_16

    return build_gap_vit_b_16(**kwargs)


@register_model
def vit_b_16_timm(**kwargs):
    # GAP version using timm's ViT-B/16 (compatible with funnybirds framework checkpoints)
    from src.models.gap_vit_b_16_timm import vit_b_16_timm as build_gap_vit_b_16_timm

    return build_gap_vit_b_16_timm(**kwargs)
