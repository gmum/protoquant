from abc import ABC, abstractmethod
from torch import nn

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
    def get_unfreezable_layers(self, model: nn.Module, num_layers: int) -> list[nn.Module]:
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
    
    def get_unfreezable_layers(self, model: nn.Module, num_layers: int) -> list[nn.Module]:
        return list(model.features[-num_layers:]) if num_layers > 0 else []

class BackboneAdapter(ModelAdapter):
    def extract_features(self, model: nn.Module) -> nn.Module:
        return model
    
    def extract_classifier(self, model: nn.Module) -> nn.Module:
        return nn.Identity()
    
    def get_unfreezable_layers(self, model: nn.Module, num_layers: int) -> list[nn.Module]:
        # For backbone models, might want to unfreeze last stages
        if hasattr(model, 'stages') and num_layers > 0:
            return list(model.stages[-num_layers:])
        return []

class ResNetAdapter(ModelAdapter):
    def extract_features(self, model: nn.Module) -> nn.Module:
        return nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4
        )
    
    def extract_classifier(self, model: nn.Module) -> nn.Module:
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            model.fc
        )
    
    def get_unfreezable_layers(self, model: nn.Module, num_layers: int) -> list[nn.Module]:
        layers = [model.layer4, model.layer3, model.layer2, model.layer1]
        return layers[:num_layers] if num_layers > 0 else []

class INaturalistResNetAdapter(ModelAdapter):
    def extract_features(self, model: ResNet_AvgPool_classifier) -> nn.Module:
        """Extract the backbone features from iNaturalist ResNet models"""
        return nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4
        )

    def extract_classifier(self, model: ResNet_AvgPool_classifier) -> nn.Module:
        """Extract the classifier head from iNaturalist ResNet models"""
        # For ResNet_AvgPool_classifier
        return nn.Sequential(
            model.avgpool,
            nn.Flatten(),
            model.classifier
        )

    def get_unfreezable_layers(self, model: nn.Module, num_layers: int) -> list[nn.Module]:
        """Get the last N layer groups that can be unfrozen for fine-tuning"""
        layers = [model.layer4, model.layer3, model.layer2, model.layer1]
        return layers[:num_layers] if num_layers > 0 else []


# Registry for adapters
MODEL_ADAPTERS: dict[str, ModelAdapter] = {
    "convnext_tiny": ConvNextAdapter(),
    "convnext_large": ConvNextAdapter(),
    "resnet50": ResNetAdapter(),
    "convnextv2_tiny_backbone": BackboneAdapter(),
    "convnextv2_nano_backbone": BackboneAdapter(),
    "inaturalist_resnet50": INaturalistResNetAdapter(),
}