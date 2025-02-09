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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        quantized = self.embeddings(indices)  # [B, H * W, C]

        # Reshape back to (B, C, H, W)
        quantized = quantized.view(batch_size, height, width, channels).permute(
            0, 3, 1, 2
        )  # [B, C, H, W]

        return quantized


def insert_codebook(
    model: nn.Module, codebook: nn.Module, model_name: str, unfreeze_before: int
) -> None:
    """Insert the codebook into the model and sets the gradient requirements

    Args:
        model (nn.Module): A PyTorch model.
        codebook (nn.Module): A module with the codebook.
        model_name (str): The name of the model.
        unfreeze_before (int): The number of layers to unfreeze before the codebook.

    Raises:
        ValueError: If the model name is not supported.
    """

    # Set requires_grad to False for all parameters
    for param in model.parameters():
        param.requires_grad = False

    if model_name == "convnext_tiny":
        if unfreeze_before > 0:
            layers_to_unfreeze = model.features[-1][-unfreeze_before:]
            logger.info(f"Unfreezing {layers_to_unfreeze}")
            for param in layers_to_unfreeze.parameters():
                param.requires_grad = True

        model.features.add_module("codebook", codebook)
    else:
        raise ValueError(f"Model {model_name} not supported")

    # Set requires_grad to True for the codebook parameters
    for param in codebook.parameters():
        param.requires_grad = True
