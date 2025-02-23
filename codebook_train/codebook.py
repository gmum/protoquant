import torch
from torch import nn
import logging

logger = logging.getLogger(__name__)

class CosineSimilarityCodebook(nn.Module):
    def __init__(self, num_entries: int, embedding_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(num_entries, embedding_dim)

        nn.init.uniform_(
            self.embeddings.weight, a=-1.0, b=1.0
        )  # Initialize codebook vectors

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate the VQ-VAE loss and quantize the input tensor using cosine similarity.

        Args:
            x (torch.Tensor): Input from the last layer.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple with the quantized tensor, codebook loss and alignment loss.
        """

        # assume the input has a batch dimension
        assert len(x.shape) == 4, "Input tensor must have a batch dimension"

        # Flatten and permute to (batch, H * W, C)
        batch_size, channels, height, width = x.shape
        x_flat = x.view(batch_size, channels, height * width).permute(
            0, 2, 1
        )  # (batch, H * W, C)

        # Normalize input vectors and codebook vectors
        x_normalized = torch.functional.F.normalize(x_flat, dim=-1)  # [B, H * W, C]
        codebook_normalized = torch.functional.F.normalize(
            self.embeddings.weight, dim=-1
        )  # [num_entries, C]

        # Compute cosine similarity between input vectors and codebook vectors
        similarity = torch.matmul(
            x_normalized, codebook_normalized.t()
        )  # [B, H * W, num_entries]

        # Find the closest codebook vector for each spatial token
        indices = torch.argmax(similarity, dim=-1)  # [B, H * W]

        # Replace each token with its closest codebook vector
        quantized_flat = self.embeddings(indices)  # [B, H * W, C]

        # Reshape back to (B, C, H, W)
        quantized = quantized_flat.view(batch_size, height, width, channels).permute(
            0, 3, 1, 2
        )  # [B, C, H, W]

        # VQ-VAE losses
        codebook_loss = torch.functional.F.mse_loss(quantized_flat.detach(), x_flat)
        commitment_loss = torch.functional.F.mse_loss(quantized_flat, x_flat.detach())

        return quantized, codebook_loss, commitment_loss


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

        codebook_wrapper = ConvNextCosineWrapper(
            features=model.features,
            codebook=codebook,
            classifier=model.classifier,
        )
    else:
        raise ValueError(f"Model {model_name} not supported")

    # Set requires_grad to True for the codebook parameters
    for param in codebook.parameters():
        param.requires_grad = True

    return codebook_wrapper


class ConvNextCosineWrapper(nn.Module):
    def __init__(self, features: nn.Module, codebook: nn.Module, classifier: nn.Module):
        super().__init__()
        self.features = features
        self.codebook = codebook
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = classifier

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.features(x)
        # Store codebook output directly without unpacking
        x, codebook_loss, commitment_loss = self.codebook(x)
        x = self.avgpool(x)
        x = self.classifier(x)

        return x, codebook_loss, commitment_loss
