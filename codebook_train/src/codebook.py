from typing import Callable
import torch
from torch import nn
import logging
from vector_quantize_pytorch import VectorQuantize
from src.custom_layers import LinearGELUNorm

logger = logging.getLogger(__name__)


class VectorQuantizeCodebook(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_entries: int,
        **kwargs,
    ):
        super().__init__()
        self.num_entries = num_entries
        self.quantizer = VectorQuantize(
            codebook_size=num_entries,
            dim=embedding_dim,
            **kwargs,
        )
        self.code_usage = torch.zeros(num_entries, dtype=torch.long)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize the vectors using the VQ-VAE codebook.

        Args:
            x (torch.Tensor): Input from the last layer.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple with the quantized tensor and the commitment loss.
        """
        input_shape = x.shape
        input_ndim = x.ndim

        if input_ndim == 4:  # CNN case: (B, C, H, W)
            # Reshape to (B*H*W, C)
            B, C, H, W = input_shape
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, C).contiguous()
        elif input_ndim == 3:  # ViT case: (B, num_tokens, depth)
            # Reshape to (B*num_tokens, depth)
            B, N, D = input_shape
            x_flat = x.reshape(-1, D).contiguous()
        else:
            raise ValueError(
                f"Unsupported input tensor rank: {input_ndim}. Must be 3 or 4."
            )

        quantized_flat, code_indices, loss = self.quantizer(x_flat)

        with torch.no_grad():
            if not self.training:
                if self.code_usage.device != x_flat.device:
                    self.code_usage = self.code_usage.to(x_flat.device)

                self.code_usage.scatter_add_(
                    0,
                    code_indices.view(-1),
                    torch.ones_like(code_indices.view(-1), dtype=torch.long),
                )

        # Reshape the quantized tensor back to the original input shape
        if input_ndim == 4:
            quantized = quantized_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        else:  # input_ndim == 3
            quantized = quantized_flat.view(B, N, D).contiguous()

        return quantized, loss

    def get_statistics(self) -> dict[str, torch.Tensor | float]:
        """Return statistics about the codebook usage.

        Returns:
            dict[str, torch.Tensor | float]: Dictionary with the statistics.
        """
        return {
            "code_usage": self.code_usage.clone(),
            "dead_ratio": (self.code_usage == 0).float().mean(),
            "median_usage": torch.median(self.code_usage.float()).item(),
            "max_usage": torch.max(self.code_usage.float()).item(),
            "min_usage": torch.min(self.code_usage.float()).item(),
        }

    def reset_statistics(self):
        self.code_usage.zero_()


class CosineSimilarityCodebook(nn.Module):
    def __init__(
        self,
        num_entries: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.embeddings = nn.Embedding(num_entries, embedding_dim)

        # Add tracking
        self.num_entries = num_entries
        self.code_usage = torch.zeros(num_entries, dtype=torch.long).to(
            self.embeddings.weight.device
        )

    def initialize_embeddings(
        self, init_func: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        """Initialize the embeddings using the provided initialization function.

        Args:
            init_func (Callable[[torch.Tensor], torch.Tensor]): Function to initialize the embeddings.
        """

        with torch.no_grad():
            init_func(self.embeddings.weight)

    def calculate_similarity(
        self, x: torch.Tensor, codes: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the cosine similarity between the input tensor and the codebook.

        Args:
            x (torch.Tensor): Input tensor.
            codes (torch.Tensor): Codebook tensor.

        Returns:
            torch.Tensor: Cosine similarity scores.
        """
        x_unit = torch.nn.functional.normalize(x, dim=-1)
        codes_unit = torch.nn.functional.normalize(codes, dim=-1)
        return x_unit @ codes_unit.t()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the VQ-VAE loss and quantize the input tensor using cosine similarity.

        Args:
            x (torch.Tensor): Input from the last layer.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple with the quantized tensor and the commitment loss.
        """

        input_shape = x.shape
        input_ndim = x.ndim

        if input_ndim == 4:  # CNN case: (B, C, H, W)
            # Reshape to (B*H*W, C)
            B, C, H, W = input_shape
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, C).contiguous()
        elif input_ndim == 3:  # ViT case: (B, num_tokens, depth)
            # Reshape to (B*num_tokens, depth)
            B, N, D = input_shape
            x_flat = x.reshape(-1, D).contiguous()
        else:
            raise ValueError(
                f"Unsupported input tensor rank: {input_ndim}. Must be 3 or 4."
            )

        with torch.no_grad():
            similarity = self.calculate_similarity(x_flat, self.embeddings.weight)
            code_indices = torch.argmax(similarity, dim=-1)

            if not self.training:
                if self.code_usage.device != x_flat.device:
                    self.code_usage = self.code_usage.to(x_flat.device)

                self.code_usage.scatter_add_(
                    0,
                    code_indices.view(-1),
                    torch.ones_like(code_indices.view(-1), dtype=torch.long),
                )

        # Quantize by selecting the closest codes
        quantized_flat = self.embeddings.weight.index_select(0, code_indices)

        # Commitment loss: encourage encoder outputs to be close to the chosen code
        commitment_loss = torch.nn.functional.mse_loss(quantized_flat, x_flat.detach())

        # Reshape the quantized tensor back to the original input shape
        if input_ndim == 4:
            quantized = quantized_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        else:  # input_ndim == 3
            quantized = quantized_flat.view(B, N, D).contiguous()

        return quantized, commitment_loss

    def get_statistics(self) -> dict[str, torch.Tensor | float]:
        """Return statistics about the codebook usage.

        Returns:
            dict[str, torch.Tensor | float]: Dictionary with the statistics.
        """
        return {
            "code_usage": self.code_usage.clone(),
            "dead_ratio": (self.code_usage == 0).float().mean(),
            "median_usage": torch.median(self.code_usage.float()).item(),
            "max_usage": torch.max(self.code_usage.float()).item(),
            "min_usage": torch.min(self.code_usage.float()).item(),
        }

    def reset_statistics(self):
        self.code_usage.zero_()


class DimReductionWrapper(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_entries: int,
        embedding_dim: int,
        in_block_config: list[int],
        out_block_config: list[int],
        mapping_dim_config: list[int],
    ):
        super().__init__()
        self.num_entries = num_entries
        self.embedding_dim = embedding_dim

        in_block = LinearGELUNorm.construct_layers([input_dim] + in_block_config)
        self.in_block = nn.Sequential(*in_block)
        self.codebook = CosineSimilarityCodebook(num_entries, embedding_dim)

        out_block = LinearGELUNorm.construct_layers([embedding_dim] + out_block_config)
        self.out_block = nn.Sequential(*out_block)

        last_dim = in_block_config[-1] if in_block_config else embedding_dim
        assert last_dim == input_dim, (
            f"Last dimension doesn't match the input dimension: {last_dim} != {input_dim}"
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize the vectors using the Cosine codebook, but with dimensionality reduction.

        Args:
            x (torch.Tensor): Input from the last layer.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple with the quantized tensor and the commitment loss.
        """

        x = self.in_block(x.permute(0, 2, 3, 1))
        quantized, commitment_loss = self.codebook(x.permute(0, 3, 1, 2))
        quantized = self.out_block(quantized.permute(0, 2, 3, 1))

        return quantized.permute(0, 3, 1, 2), commitment_loss

    def get_statistics(self) -> dict[str, torch.Tensor | float]:
        """Return statistics about the codebook usage.

        Returns:
            dict[str, torch.Tensor | float]: Dictionary with the statistics.
        """
        return self.codebook.get_statistics()

    def reset_statistics(self):
        self.codebook.reset_statistics()
