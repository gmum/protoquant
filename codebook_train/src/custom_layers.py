import torch
import torch.nn as nn


class LinearGELUNorm(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.activation(x)
        x = self.norm(x)
        return x

    @classmethod
    def construct_layers(cls, dim_config: list[int]) -> list[nn.Module]:
        """Construct a list of LinearGELUNorm layers based on the given configuration.

        Args:
            dim_config (list[int]): List of integers representing the dimensions of each layer.

        Returns:
            list[nn.Module]: List of layers.
        """

        layers = []

        for i in range(len(dim_config) - 1):
            layers.append(LinearGELUNorm(dim_config[i], dim_config[i + 1]))

        return layers
