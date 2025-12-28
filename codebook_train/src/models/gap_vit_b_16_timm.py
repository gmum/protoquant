import timm
from torch import nn


def vit_b_16_timm(num_classes: int = 1000, pretrained: bool = True, **kwargs) -> nn.Module:
    """
    Build a GAP-based ViT-B/16 using timm's implementation.
    
    This uses timm's vit_base_patch16_224 model with global_pool='avg',
    which applies Global Average Pooling over patch tokens (compatible with
    the funnybirds framework checkpoints).
    
    Args:
        num_classes: Output dimension for the classifier head.
        pretrained: Whether to load pretrained ImageNet weights.
        **kwargs: Forwarded to timm.create_model.
    
    Returns:
        nn.Module: A ViT-B/16 model with GAP pooling.
    """
    model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=pretrained,
        num_classes=num_classes,
        global_pool='avg',
        **kwargs
    )
    return model
