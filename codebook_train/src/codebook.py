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
        self.register_buffer("code_usage", torch.zeros(num_entries, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize the vectors using the VQ-VAE codebook.

        Args:
            x (torch.Tensor): Input from the last layer.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple with the quantized tensor and the commitment loss.
        """
        assert len(x.shape) == 4

        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)

        quantized, indices, loss = self.quantizer(x)
        quantized = quantized.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        self.code_usage += torch.bincount(
            indices.view(-1), minlength=self.num_entries
        ).long()

        return quantized, loss

    def get_statistics(self) -> dict[str, torch.Tensor]:
        """Return statistics about the codebook usage.

        Returns:
            dict[str, torch.Tensor]: Dictionary with the statistics.
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
        mapping_dim_config: list[int],
        entropy_loss_weight: float,
    ):
        super().__init__()
        self.entropy_weight = entropy_loss_weight
        self.embeddings = nn.Embedding(num_entries, embedding_dim)

        mapping_layers = LinearGELUNorm.construct_layers(
            [embedding_dim] + mapping_dim_config
        )
        self.codebook_mapping = nn.Sequential(*mapping_layers)

        # Add tracking buffers
        self.num_entries = num_entries
        self.register_buffer("code_usage", torch.zeros(num_entries, dtype=torch.long))
        self.register_buffer(
            "restarted_count", torch.zeros(num_entries, dtype=torch.long)
        )
        
    def initialize_embeddings(self, init_func: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Initialize the embeddings using the provided initialization function.
        
        Args:
            init_func (Callable[[torch.Tensor], torch.Tensor]): Function to initialize the embeddings.
        """
        
        with torch.no_grad():
            init_func(self.embeddings.weight)
        

    def calculate_entropy_loss(self, code_indices: torch.Tensor) -> torch.Tensor:
        """Calculate entropy loss to encourage uniform codebook usage.

        Args:
            code_indices (torch.Tensor): Indices of selected codes for the batch

        Returns:
            torch.Tensor: Entropy loss
        """
        # Calculate histogram of code usage in current batch
        batch_histogram = torch.bincount(
            code_indices.view(-1), minlength=self.num_entries
        ).float()

        # Convert to probability distribution (normalize)
        batch_probs = batch_histogram / (batch_histogram.sum() + 1e-8)

        # Calculate entropy: -sum(p_i * log(p_i))
        # Adding small epsilon to avoid log(0)
        entropy = -torch.sum(batch_probs * torch.log(batch_probs + 1e-8))

        # We want to maximize entropy, so return negative entropy as loss
        # Normalize by log(num_entries) to keep loss in reasonable range
        max_entropy = torch.log(
            torch.tensor(self.num_entries, dtype=torch.float, device=batch_probs.device)
        )
        entropy_loss = 1.0 - (
            entropy / max_entropy
        )  # Normalized [0,1] where 0 is uniform distribution

        return entropy_loss

    def restart_codes(self, restart_threshold: int) -> None:
        """Restart the codebook by reinitializing the embeddings under the threshold.

        Args:
            restart_threshold (int): The threshold for restarting the codebook.
        """

        # Restart the codebook by reinitializing the embeddings under the threshold
        to_restart = self.code_usage <= restart_threshold
        to_restart_indices = torch.nonzero(to_restart, as_tuple=False).squeeze()

        if len(to_restart_indices) == 0:
            return

        # Get the most used codes to draw inspiration from
        active_indices = (-self.code_usage).argsort()[
            : self.num_entries // 2
        ]  # Top 100 most used codes
        # Take inspiration from successful codes but add noise
        sample_indices = torch.randint(
            0, len(active_indices), (len(to_restart_indices),)
        )
        source_embeddings = self.embeddings.weight[active_indices[sample_indices]]

        # Add noise proportional to the magnitude of the embeddings
        noise_scale = 0.1 * torch.norm(source_embeddings, dim=1, keepdim=True)
        noise = torch.randn_like(source_embeddings) * noise_scale
        new_embeddings = source_embeddings + noise

        # Update the dead codes with the new embeddings
        self.embeddings.weight.data[to_restart_indices] = new_embeddings
        self.restarted_count[to_restart_indices] += 1

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
        x_unit = torch.functional.F.normalize(x, dim=-1)
        codes_unit = torch.functional.F.normalize(codes, dim=-1)
        return x_unit @ codes_unit.t()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the VQ-VAE loss and quantize the input tensor using cosine similarity.

        Args:
            x (torch.Tensor): Input from the last layer.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple with the quantized tensor and the commitment loss.
        """

        assert len(x.shape) == 4

        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)

        with torch.no_grad():
            mapped_codes = self.codebook_mapping(self.embeddings.weight)
            similarity = self.calculate_similarity(x, mapped_codes)
            code_indices = torch.argmax(similarity, dim=-1)

            # sum over batch dim
            count = torch.bincount(code_indices.view(-1), minlength=self.num_entries)
            self.code_usage += count.long()

        quantized = self.codebook_mapping(self.embeddings(code_indices))
        commitment_loss = torch.functional.F.mse_loss(quantized, x.detach())
        quantized = quantized.view(B, H, W, C).permute(0, 3, 1, 2)

        # Calculate entropy loss
        entropy_loss = self.calculate_entropy_loss(code_indices)
        codebook_loss = commitment_loss + self.entropy_weight * entropy_loss

        return quantized, codebook_loss

    def get_statistics(self) -> dict[str, torch.Tensor]:
        """Return statistics about the codebook usage.

        Returns:
            dict[str, torch.Tensor]: Dictionary with the statistics.
        """
        return {
            "code_usage": self.code_usage.clone(),
            "dead_ratio": (self.code_usage == 0).float().mean(),
            "median_usage": torch.median(self.code_usage.float()).item(),
            "max_usage": torch.max(self.code_usage.float()).item(),
            "min_usage": torch.min(self.code_usage.float()).item(),
            "restarted_count": self.restarted_count.clone(),
            "restart_ratio": (self.restarted_count > 0).float().mean(),
        }

    def reset_statistics(self):
        self.code_usage.zero_()
        self.restarted_count.zero_()


class DimReductionWrapper(nn.Module):

    def __init__(
        self,
        input_dim: int,
        num_entries: int,
        embedding_dim: int,
        in_block_config: list[int],
        out_block_config: list[int],
        mapping_dim_config: list[int],
        entropy_loss_weight: float,
    ):
        super().__init__()
        self.num_entries = num_entries
        self.embedding_dim = embedding_dim

        in_block = LinearGELUNorm.construct_layers([input_dim] + in_block_config)
        self.in_block = nn.Sequential(*in_block)
        self.codebook = CosineSimilarityCodebook(
            num_entries, embedding_dim, mapping_dim_config, entropy_loss_weight
        )

        out_block = LinearGELUNorm.construct_layers([embedding_dim] + out_block_config)
        self.out_block = nn.Sequential(*out_block)

        last_dim = in_block_config[-1] if in_block_config else embedding_dim
        assert (
            last_dim == input_dim
        ), f"Last dimension doesn't match the input dimension: {last_dim} != {input_dim}"

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

    def get_statistics(self) -> dict[str, torch.Tensor]:
        """Return statistics about the codebook usage.

        Returns:
            dict[str, torch.Tensor]: Dictionary with the statistics.
        """
        return self.codebook.get_statistics()

    def reset_statistics(self):
        self.codebook.reset_statistics()

    def restart_codes(self, restart_threshold: int) -> None:
        """Restart the codebook by reinitializing the embeddings under the threshold.

        Args:
            restart_threshold (int): The threshold for restarting the codebook.
        """

        self.codebook.restart_codes(restart_threshold)
