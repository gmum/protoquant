from torchvision import models


MODELS = {}


def register_model(func):
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
