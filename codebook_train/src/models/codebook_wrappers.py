import torch
from torch import nn
import logging

from src.models.adapters import MODEL_ADAPTERS

logger = logging.getLogger(__name__)


class CNNCodebookWrapper(nn.Module):
    def __init__(self, features: nn.Module, codebook: nn.Module, classifier: nn.Module):
        """
        Wrapper for a CNN model with a codebook and classifier.
        This wrapper allows for separation of codebook and classifier loss calculations.

        Args:
            features (nn.Module): Feature extractor module.
            codebook (nn.Module): Codebook module.
            classifier (nn.Module): Classifier module.
        """

        super().__init__()
        self.features = features
        self.codebook = codebook
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        x, codebook_loss = self.codebook(x)
        x = self.classifier(x)

        return x, codebook_loss


def create_codebook_wrapper(
    model: nn.Module, 
    codebook: nn.Module, 
    model_name: str, 
    unfreeze_before: int
) -> CNNCodebookWrapper:
    """Insert the codebook into the model using model-specific adapters"""
    
    if model_name not in MODEL_ADAPTERS:
        raise ValueError(f"Model {model_name} not supported. Available: {list(MODEL_ADAPTERS.keys())}")
    
    adapter = MODEL_ADAPTERS[model_name]
    
    # Set all parameters to not require gradients
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze specific layers if requested
    if unfreeze_before > 0:
        layers_to_unfreeze = adapter.get_unfreezable_layers(model, unfreeze_before)
        logger.info(f"Unfreezing {len(layers_to_unfreeze)} layer groups")
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True
    
    # Create wrapper using adapter
    codebook_wrapper = CNNCodebookWrapper(
        features=adapter.extract_features(model),
        codebook=codebook,
        classifier=adapter.extract_classifier(model),
    )
    
    # Enable gradients for codebook
    for param in codebook.parameters():
        param.requires_grad = True
    
    return codebook_wrapper
