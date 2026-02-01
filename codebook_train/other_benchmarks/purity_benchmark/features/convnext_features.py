import torch
import torch.nn as nn
from torchvision import models


def _convnext_tiny(pretrained: bool) -> nn.Module:
    # torchvision deprecated the old `pretrained=` flag; `weights=None` is the
    # correct way to request random init.
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    return models.convnext_tiny(weights=weights)


def replace_convlayers_convnext(model, threshold):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_convlayers_convnext(module, threshold)
        if isinstance(module, nn.Conv2d):
            if module.stride[0] == 2:
                if (
                    module.in_channels > threshold
                ):  # replace bigger strides to reduce receptive field, skip some 2x2 layers. >100 gives output size (26, 26). >300 gives (13, 13)
                    module.stride = tuple(s // 2 for s in module.stride)

    return model


def convnext_tiny_26_features(pretrained=False, **kwargs):
    model = _convnext_tiny(pretrained)
    with torch.no_grad():
        model.avgpool = nn.Identity()
        model.classifier = nn.Identity()
        model = replace_convlayers_convnext(model, 100)

    return model


def convnext_tiny_13_features(pretrained=False, **kwargs):
    model = _convnext_tiny(pretrained)
    with torch.no_grad():
        model.avgpool = nn.Identity()
        model.classifier = nn.Identity()
        model = replace_convlayers_convnext(model, 300)

    return model


def convnext_tiny_7_features(pretrained=False, **kwargs):
    """Torchvision ConvNeXt-Tiny with original stride (224 -> ~7x7 latent grid).

    Unlike convnext_tiny_13 / convnext_tiny_26, this does not modify Conv2d
    strides. It is intended for fair comparisons against methods using the
    standard backbone resolution.
    """
    model = _convnext_tiny(pretrained)
    with torch.no_grad():
        model.avgpool = nn.Identity()
        model.classifier = nn.Identity()
    return model
