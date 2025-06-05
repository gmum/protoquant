import torch
from torch import nn
import logging

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
    model: nn.Module, codebook: nn.Module, model_name: str, unfreeze_before: int
) -> nn.Module:
    """Insert the codebook into the model and sets the gradient requirements

    Args:
        model (nn.Module): A PyTorch model.
        codebook (nn.Module): A module with the codebook.
        model_name (str): The name of the model.
        unfreeze_before (int): The number of layers to unfreeze before the codebook.

    Raises:
        ValueError: If the model name is not supported.

    Returns:
        nn.Module: A wrapper module with the codebook inserted.
    """

    # Set requires_grad to False for all parameters
    for param in model.parameters():
        param.requires_grad = False

    if model_name == "convnext_tiny":
        if unfreeze_before > 0:
            layers_to_unfreeze = model.features[-unfreeze_before:]
            logger.info(f"Unfreezing {layers_to_unfreeze}")
            for param in layers_to_unfreeze.parameters():
                param.requires_grad = True

        codebook_wrapper = CNNCodebookWrapper(
            features=model.features,
            codebook=codebook,
            classifier=nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                model.classifier,
            ),
        )
    elif model_name.endswith("backbone"):
        codebook_wrapper = CNNCodebookWrapper(
            features=model,
            codebook=codebook,
            classifier=nn.Identity(),
        )
    elif model_name == "head_only":
        codebook_wrapper = CNNCodebookWrapper(
            features=nn.Identity(),
            codebook=codebook,
            classifier=nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                model.classifier,
            ),
        )
    else:
        raise ValueError(f"Model {model_name} not supported")

    # Set requires_grad to True for the codebook parameters
    for param in codebook.parameters():
        param.requires_grad = True

    return codebook_wrapper
