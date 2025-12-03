import torch
from torch import nn
from torchvision import models

class ViTWithGAP(nn.Module):
    """
    GAP wrapper for torchvision VisionTransformer (ViT).
    - Patch embed via _process_input (returns patch tokens, no CLS).
    - Prepend CLS, run encoder.
    - Global Average Pool over patch tokens (exclude CLS).
    - Classifier head on pooled representation.
    Exposes forward_features / forward_head so generic adapters can use it.
    """
    def __init__(self, vit: nn.Module):
        super().__init__()
        self.vit = vit

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Returns full token sequence including CLS after encoder
        x = self.vit._process_input(x)              # (B, N_patches, C)
        n = x.shape[0]
        cls = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat((cls, x), dim=1)              # (B, 1+N_patches, C)
        x = self.vit.encoder(x)                     # (B, 1+N_patches, C)
        return x

    def forward_head(self, tokens: torch.Tensor) -> torch.Tensor:
        gap = tokens[:, 1:].mean(dim=1)             # (B, C)
        return self.vit.heads(gap)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_features(x)
        return self.forward_head(tokens)

    def reset_classifier(self, num_classes: int, global_pool: str = "") -> None:
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)

def vit_b_16(num_classes: int = 1000, weights=None, **kwargs) -> nn.Module:
    """
    Build a GAP-based ViT-B/16 on top of torchvision's implementation.
    - Loads torchvision.models.vit_b_16 (optionally with pretrained weights).
    - Replaces the classifier to match num_classes.
    - Wraps the model with ViTWithGAP which applies GAP over patch tokens (excludes CLS).
    
    Args:
        num_classes: Output dimension for the classifier head.
        weights: Optional torchvision.models.ViT_B_16_Weights for pretrained loading.
        **kwargs: Forwarded to torchvision.models.vit_b_16 (e.g., progress).
    Returns:
        nn.Module: ViTWithGAP-wrapped ViT-B/16 model.
    """
    # Forward all kwargs except accidental num_classes (we control it here)
    base = models.vit_b_16(weights=weights, **{k: v for k, v in kwargs.items() if k != "num_classes"})
    in_features = base.heads.head.in_features
    base.heads.head = nn.Linear(in_features, num_classes)
    return ViTWithGAP(base)
