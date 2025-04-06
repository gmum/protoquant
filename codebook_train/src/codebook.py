import torch
from torch import nn
import logging

logger = logging.getLogger(__name__)

class CosineSimilarityCodebook(nn.Module):
    def __init__(self, num_entries: int, embedding_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(num_entries, embedding_dim)
        nn.init.orthogonal_(
            self.embeddings.weight,
        )

        # Add tracking buffers
        self.num_entries = num_entries
        self.register_buffer("code_usage", torch.zeros(num_entries, dtype=torch.long))

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
            c = self.embeddings.weight

            x_unit = torch.functional.F.normalize(x, dim=-1)
            c_unit = torch.functional.F.normalize(c, dim=-1)

            sim = x_unit @ c_unit.t()
            indices = torch.argmax(sim, dim=-1)

            if self.training:
                # sum over batch dim
                count = torch.bincount(indices.view(-1), minlength=self.num_entries)
                self.code_usage += count.long()

        quantized = self.embeddings(indices)

        commitment_loss = torch.functional.F.mse_loss(quantized, x.detach())
        quantized = quantized.view(B, H, W, C).permute(0, 3, 1, 2)

        return quantized, commitment_loss

    def get_statistics(self) -> dict[str, torch.Tensor]:
        """Return statistics about the codebook usage.

        Returns:
            dict[str, torch.Tensor]: Dictionary with the statistics.
        """
        return {
            "code_usage": self.code_usage.clone(),
            "dead_ratio": (self.code_usage == 0).float().mean(),
        }

    def reset_statistics(self):
        self.code_usage.zero_()


class DimReductionWrapper(nn.Module):

    def __init__(
        self,
        input_dim: int,
        num_entries: int,
        embedding_dim: int,
        normalize: bool,
        activation: str,
    ):
        super().__init__()
        self.num_entries = num_entries
        self.embedding_dim = embedding_dim

        self.downsample = nn.Linear(input_dim, embedding_dim)
        self.upsample = nn.Linear(embedding_dim, input_dim)
        self.codebook = CosineSimilarityCodebook(num_entries, embedding_dim)

        self.normalize = normalize
        self.input_norm = nn.LayerNorm(embedding_dim)
        self.output_norm = nn.LayerNorm(input_dim)

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "identity":
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Activation {activation} not supported")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize the vectors using the Cosine codebook, but with dimensionality reduction.

        Args:
            x (torch.Tensor): Input from the last layer.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple with the quantized tensor and the commitment loss.
        """

        x_downsampled = self.downsample(x.permute(0, 2, 3, 1))
        if self.normalize:
            x_downsampled = self.input_norm(x_downsampled)

        quantized, commitment_loss = self.codebook(x_downsampled.permute(0, 3, 1, 2))
        quantized_upsampled = self.upsample(quantized.permute(0, 2, 3, 1))
        quantized_upsampled = self.activation(quantized_upsampled)

        if self.normalize:
            quantized_upsampled = self.output_norm(quantized_upsampled)

        return quantized_upsampled.permute(0, 3, 1, 2), commitment_loss

    def get_statistics(self) -> dict[str, torch.Tensor]:
        """Return statistics about the codebook usage.

        Returns:
            dict[str, torch.Tensor]: Dictionary with the statistics.
        """
        return self.codebook.get_statistics()

    def reset_statistics(self):
        self.codebook.reset_statistics()

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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        # Store codebook output directly without unpacking
        x, commitment_loss = self.codebook(x)
        x = self.avgpool(x)
        x = self.classifier(x)

        return x, commitment_loss
