import torch
from torch import nn


class Codebook(nn.Module):
    def __init__(self, num_entries: int, embedding_dim: int):
        super(Codebook, self).__init__()
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
