from abc import ABC, abstractmethod
from torch import nn
import torch
from src.models.inaturalist_resnet import ResNet_AvgPool_classifier


class ModelAdapter(ABC):
    """Abstract base class for model-specific adapters"""

    @abstractmethod
    def extract_features(self, model: nn.Module) -> nn.Module:
        """Extract the feature extractor from the model"""
        pass

    @abstractmethod
    def extract_classifier(self, model: nn.Module) -> nn.Module:
        """Extract the classifier from the model"""
        pass

    @abstractmethod
    def get_unfreezable_layers(
        self, model: nn.Module, num_layers: int
    ) -> list[nn.Module]:
        """Get the last N layers that can be unfrozen"""
        pass


class ConvNextAdapter(ModelAdapter):
    def extract_features(self, model: nn.Module) -> nn.Module:
        return model.features

    def extract_classifier(self, model: nn.Module) -> nn.Module:
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            model.classifier,
        )

    def get_unfreezable_layers(
        self, model: nn.Module, num_layers: int
    ) -> list[nn.Module]:
        return list(model.features[-num_layers:]) if num_layers > 0 else []


class BackboneAdapter(ModelAdapter):
    def extract_features(self, model: nn.Module) -> nn.Module:
        return model

    def extract_classifier(self, model: nn.Module) -> nn.Module:
        return nn.Identity()

    def get_unfreezable_layers(
        self, model: nn.Module, num_layers: int
    ) -> list[nn.Module]:
        # For backbone models, might want to unfreeze last stages
        if hasattr(model, "stages") and num_layers > 0:
            return list(model.stages[-num_layers:])
        return []


class ResNetAdapter(ModelAdapter):
    def extract_features(self, model: nn.Module) -> nn.Module:
        return nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )

    def extract_classifier(self, model: nn.Module) -> nn.Module:
        return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), model.fc)

    def get_unfreezable_layers(
        self, model: nn.Module, num_layers: int
    ) -> list[nn.Module]:
        layers = [model.layer4, model.layer3, model.layer2, model.layer1]
        return layers[:num_layers] if num_layers > 0 else []


class INaturalistResNetAdapter(ModelAdapter):
    def extract_features(self, model: ResNet_AvgPool_classifier) -> nn.Module:
        """Extract the backbone features from iNaturalist ResNet models"""
        return nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )

    def extract_classifier(self, model: ResNet_AvgPool_classifier) -> nn.Module:
        """Extract the classifier head from iNaturalist ResNet models"""
        # For ResNet_AvgPool_classifier
        return nn.Sequential(model.avgpool, nn.Flatten(), model.classifier)

    def get_unfreezable_layers(
        self, model: nn.Module, num_layers: int
    ) -> list[nn.Module]:
        """Get the last N layer groups that can be unfrozen for fine-tuning"""
        layers = [model.layer4, model.layer3, model.layer2, model.layer1]
        return layers[:num_layers] if num_layers > 0 else []


class _ViTFeaturesWrapper(nn.Module):
    """Wraps the ViT's forward_features method."""

    def __init__(self, vit_model: nn.Module):
        super().__init__()
        self.vit_model = vit_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This will return the sequence of token embeddings
        return self.vit_model.forward_features(x)


class _ViTHeadWrapper(nn.Module):
    """Wraps the ViT's forward_head method."""

    def __init__(self, vit_model: nn.Module):
        super().__init__()
        self.vit_model = vit_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This takes the token embeddings and returns classification logits
        return self.vit_model.forward_head(x)


# --- Vision Transformer Adapter Implementation ---


class VisionTransformerAdapter(ModelAdapter):
    """Adapter for the timm VisionTransformer model."""

    def extract_features(self, model: nn.Module) -> nn.Module:
        """
        Extracts the feature extractor part of the Vision Transformer.
        This includes patch embedding, position embedding, and transformer blocks.
        """
        # We wrap the model's own forward_features method in a nn.Module
        return _ViTFeaturesWrapper(model)

    def extract_classifier(self, model: nn.Module) -> nn.Module:
        """
        Extracts the classifier head of the Vision Transformer.
        This includes global pooling and the final linear layer.
        """
        # We wrap the model's own forward_head method in a nn.Module
        return _ViTHeadWrapper(model)

    def get_unfreezable_layers(
        self, model: nn.Module, num_layers: int
    ) -> list[nn.Module]:
        """
        Get the last N layer groups for fine-tuning. For ViT, this typically
        means unfreezing from the head backwards.

        Order of unfreezing: head -> fc_norm -> final norm -> transformer blocks (last to first)
        """
        if num_layers <= 0:
            return []

        # Create a list of layers from the end of the model to the beginning
        unfreezable = [model.head, model.fc_norm, model.norm]

        # Add transformer blocks in reverse order
        unfreezable.extend(list(reversed(model.blocks)))

        # Return the first `num_layers` from this list
        return unfreezable[:num_layers]


# --- TorchVision ViT (GAP wrapper) Adapter ---


class TorchVisionViTAdapter(ModelAdapter):
    """
    Adapter for ViTWithGAP (torchvision backbone wrapped for GAP).
    Unfreezing order: heads -> encoder layers (last to first).
    """

    def extract_features(self, model: nn.Module) -> nn.Module:
        return _ViTFeaturesWrapper(model)

    def extract_classifier(self, model: nn.Module) -> nn.Module:
        return _ViTHeadWrapper(model)

    def get_unfreezable_layers(
        self, model: nn.Module, num_layers: int
    ) -> list[nn.Module]:
        if num_layers <= 0:
            return []
        layers: list[nn.Module] = [model.vit.heads]
        # encoder.layers is an Iterable of TransformerEncoderLayer
        if hasattr(model.vit, "encoder") and hasattr(model.vit.encoder, "layers"):
            enc_layers = list(model.vit.encoder.layers)
            layers.extend(reversed(enc_layers))
        return layers[:num_layers]


def guess_adapter(model: nn.Module) -> ModelAdapter:
    """
    Guess the appropriate model adapter based on the model.

    Args:
        model (nn.Module): The model instance.

    Returns:
        ModelAdapter: The appropriate model adapter.
    """

    # Prefer TorchVision GAP ViT first so we don't fall back to the generic ViT adapter
    if model.__class__.__name__ == "ViTWithGAP" or (
        hasattr(model, "vit") and hasattr(model.vit, "encoder")
    ):
        return TorchVisionViTAdapter()

    if hasattr(model, "features") and hasattr(model, "classifier"):
        return ConvNextAdapter()

    # Generic ViT (e.g., timm): exposes forward_features/forward_head on the base model
    if hasattr(model, "forward_features") and hasattr(model, "forward_head"):
        return VisionTransformerAdapter()

    if hasattr(model, "fc") and hasattr(model, "conv1"):
        return ResNetAdapter()

    if hasattr(model, "classifier") and hasattr(model, "conv1"):
        return INaturalistResNetAdapter()

    raise ValueError(
        f"Can't guess the adapter for the model: {model.__class__.__name__}"
    )
