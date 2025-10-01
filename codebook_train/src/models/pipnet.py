import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PIPNetOutput:
    proto_fmap: Tensor | None = None
    proto_fvec: Tensor | None = None
    logits: Tensor | None = None
    pre_softmax: Tensor | None = None
    softmax_out: Tensor | None = None
    proto_sim: Tensor | None = None


class NonNegLinear(nn.Module):
    def __init__(
        self,
        fan_in: int,
        fan_out: int,
        bias: bool = True,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty((fan_out, fan_in)),
        )
        nn.init.normal_(self.weight, mean=1.0, std=0.1)
        self.weight.requires_grad = True

        self.normalization_multiplier = nn.Parameter(
            torch.ones((1,), requires_grad=True),
        )

        nn.init.constant_(self.normalization_multiplier, val=2.0)
        if bias:
            self.bias = nn.Parameter(torch.zeros(fan_out))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, torch.relu(self.weight), self.bias)


class QuantizedPIPNetHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        codebook: torch.Tensor,
        normalize: bool = True,
        bias: bool = False,
        train_codebook: bool = False,
        temperature: float = 0.1
    ):
        super().__init__()
        self.K = num_classes
        self.P, self.D = codebook.shape
        self.normalize = normalize
        self.temperature = temperature

        if train_codebook:
            self.codebook = nn.Parameter(codebook, requires_grad=True)
            logger.info("INFO: Codebook registered as a trainable nn.Parameter.")
        else:
            self.register_buffer("codebook", codebook)
            logger.info("INFO: Codebook registered as a non-trainable buffer.")

        # PIP-Net pool;
        self.pool_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=(1, 1)),
            nn.Flatten(),
        )

        # PIP-Net classifier;
        self.classifier = NonNegLinear(
            fan_in=self.P,
            fan_out=self.K,
            bias=bias,
        )

    @property
    def multiplier(self):
        return self.classifier.normalization_multiplier
    
    def forward(self, x: Tensor) -> PIPNetOutput:
        sim_map = self.detect_protos(x)
        pooled_scores = self.pool(F.softmax(sim_map / self.temperature, dim=1))
        logits = self.classifier(pooled_scores)

        # Return all the necessary tensors for analysis
        return PIPNetOutput(
            proto_fmap=sim_map, # The crucial (B, P, H, W) similarity map
            proto_fvec=pooled_scores, # The (B, P) pooled similarity vector
            logits=logits
        )


    def detect_protos(self, x):
        input_ndim = x.ndim
        if input_ndim == 4:  # CNN case: (B, C, H, W)
            B, C, H, W = x.shape
            # Reshape to (B*H*W, C)
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
        elif input_ndim == 3:  # ViT case: (B, N, D)
            B, N, D = x.shape
            # If there's a class token, remove it.
            if N == 197 or N == 257: # Common for 224x224 and 256x256 ViTs
                logger.debug(f"Removing CLS token from ViT features. Original num_tokens: {N}")
                x = x[:, 1:, :]
                N = N - 1
            
            # Reshape to (B*N, D)
            x_flat = x.reshape(-1, D)
            
            # Calculate pseudo-height/width for reshaping later
            if N == 196:
                side_len = 14
            elif N == 256:
                side_len = 16
            else:
                raise ValueError(f"Unsupported number of tokens for ViT features: {N}")

            if side_len * side_len != N:
                raise ValueError(
                    f"For ViT features, the number of tokens ({N}) must be a perfect square "
                    "to be reshaped into a 2D map."
                )
            H, W = side_len, side_len
        else:
            raise ValueError(f"Unsupported input tensor rank: {input_ndim}. Must be 3 or 4.")

        # --- Core similarity calculation (same for both cases) ---
        c = self.codebook
        
        if self.normalize:
            epsilon = 1e-6
            x_norm = x_flat.norm(dim=-1, keepdim=True)
            c_norm = c.norm(dim=-1, keepdim=True)
            x_unit = x_flat / (x_norm + epsilon)
            c_unit = c / (c_norm + epsilon)
            sim = x_unit @ c_unit.t() # Cosine similarity
        else:
            # L2 distance squared, shaped as a similarity (higher is better)
            # -||x-c||^2 = - (x^2 - 2xc + c^2)
            # We can ignore x^2 and c^2 as they don't change the argmax
            sim = x_flat @ c.t()

        # --- Reshape back to a spatial map ---
        # `sim` has shape (B * num_vectors, P)
        sim = sim.view(B, H, W, self.P).permute(0, 3, 1, 2)
        return sim

    def pool(self, x: Tensor) -> Tensor:
        return self.pool_layer(x)

    # def zero_out_irrelevant(self, x: Tensor) -> Tensor:
    #    return torch.where(proto_fvec < 0.1, 0.0, proto_fvec)

    def pre_softmax_norm(self, x: Tensor) -> Tensor:
        return torch.log1p(x)  # ** self.multiplier)
