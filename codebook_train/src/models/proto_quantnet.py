from dataclasses import dataclass
from typing import Optional
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ProtoQuantOutput:
    """Output from ProtoQuantNet forward pass.
    
    Attributes:
        logits: Class predictions (B, num_classes)
        pooled_proto_scores: Pooled prototype activations (B, num_prototypes)
        similarity_map: Spatial prototype activations (B, P, H, W), optional
        classifier_sparsity_loss: Sparsity regularization on classifier weights, optional
    """

    logits: torch.Tensor
    pooled_proto_scores: torch.Tensor
    similarity_map: Optional[torch.Tensor] = None
    classifier_sparsity_loss: Optional[torch.Tensor] = None


class PrototypeClassifier(nn.Module):
    """Prototype-based classifier with non-negative weights for interpretability.
    
    Args:
        fan_in (int): Input dimension.
        fan_out (int): Output dimension.
        bias (bool): Whether to include bias term. Defaults to False.
    """
    
    def __init__(self, fan_in: int, fan_out: int, bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((fan_out, fan_in)))
        nn.init.normal_(self.weight, mean=1.0, std=0.1)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(fan_out))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, torch.relu(self.weight), self.bias)


class ProtoQuantNet(nn.Module):
    """Unified quantized prototype network combining backbone, codebook, and classifier.
    
    Args:
        backbone (nn.Module): Feature extraction network (CNN or ViT).
        codebook (CodebookBase): Codebook module satisfying CodebookBase.
        num_classes (int): Number of output classes.
        temperature (float): Temperature for softmax pooling. Defaults to 0.1.
        bias (bool): Use bias in classifier layer. Defaults to False.
        classifier_sparsity_lambda (float): Regularization strength to encourage sparse
            prototype usage in classifier. Set to 0 to disable. Defaults to 0.
    """
    
    _EPSILON: float = 1e-6  # Numerical stability constant
    
    def __init__(
        self,
        backbone: nn.Module,
        codes: torch.Tensor,
        num_classes: int,
        train_codebook: bool = False,
        temperature: float = 0.1,
        bias: bool = False,
        classifier_sparsity_lambda: float = 0.0,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        
        self.backbone = backbone
        self.num_classes = num_classes
        self.temperature = temperature
        self.classifier_sparsity_lambda = classifier_sparsity_lambda
        
        self.pool_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=(1, 1)),
            nn.Flatten(),
        )

        if train_codebook:
            self.codes = nn.Parameter(codes, requires_grad=True)
            logger.info("Codebook registered as a trainable nn.Parameter.")
        else:
            self.register_buffer("codes", codes)
            logger.info("Codebook registered as a non-trainable buffer.")

        self.num_prototypes = codes.shape[0]
        self.classifier = PrototypeClassifier(
            fan_in=self.num_prototypes,
            fan_out=num_classes,
            bias=bias,
        )

        if freeze_backbone:
            logger.info("Freezing backbone parameters.")
            for p in self.backbone.parameters():
                p.requires_grad = False

        logger.info(f"ProtoQuantNet: {self.num_prototypes} prototypes, {num_classes} classes, "
                   f"sparsity lambda: {classifier_sparsity_lambda}")

    def forward(self, x: torch.Tensor, return_similarity_map: bool = False) -> ProtoQuantOutput:
        """Forward pass through backbone, prototype detection, and classification.
        
        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).
            return_similarity_map (bool): Whether to return spatial similarity map.
                Defaults to False.
        
        Returns:
            ProtoQuantOutput: Output containing logits, pooled_proto_scores,
                and optionally similarity_map and classifier_sparsity_loss.
        """
        features = self.backbone(x)
        similarity_map = self._detect_prototypes(features)
        pooled_proto_scores = self.pool_layer(F.softmax(similarity_map / self.temperature, dim=1))
        logits = self.classifier(pooled_proto_scores)
        
        # Compute L1 sparsity loss on classifier weights if enabled
        classifier_sparsity_loss = None
        if self.classifier_sparsity_lambda > 0:
            positive_weights = torch.relu(self.classifier.weight)
            classifier_sparsity_loss = self.classifier_sparsity_lambda * positive_weights.sum()
        
        return ProtoQuantOutput(
            logits=logits,
            pooled_proto_scores=pooled_proto_scores,
            similarity_map=similarity_map if return_similarity_map else None,
            classifier_sparsity_loss=classifier_sparsity_loss,
        )

    def _detect_prototypes(self, x: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between features and codebook prototypes.
        
        Args:
            x (torch.Tensor): Feature tensor of shape (B, C, H, W) or (B, N, D).
        
        Returns:
            torch.Tensor: Similarity map of shape (B, num_prototypes, H, W).
        """
        input_ndim = x.ndim
        
        if input_ndim == 4:
            # CNN features: (B, C, H, W)
            B, C, H, W = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
        elif input_ndim == 3:
            # ViT features: (B, N, D)
            B, N, D = x.shape
            if N in (197, 257):
                # Remove CLS token for ViT
                x = x[:, 1:, :]
                N -= 1
            x_flat = x.reshape(-1, D)
            
            # Infer spatial dimensions from token count
            side_len = int(N ** 0.5)
            if side_len * side_len != N:
                raise ValueError(f"ViT features must have square number of tokens, got {N}")
            H = W = side_len
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {input_ndim}D")

        # Compute cosine similarity between features and prototypes
        x_unit = x_flat / (x_flat.norm(dim=-1, keepdim=True) + self._EPSILON)
        c_unit = self.codes / (self.codes.norm(dim=-1, keepdim=True) + self._EPSILON)
        similarity = x_unit @ c_unit.t()
        
        return similarity.view(B, H, W, self.num_prototypes).permute(0, 3, 1, 2)

    def calculate_local_size(self, threshold: float = 0.1) -> torch.Tensor:
        """Calculate number of prototypes with weight > threshold for each class.
        
        Args:
            threshold (float): Minimum weight to count a prototype. Defaults to 0.1.
        
        Returns:
            torch.Tensor: Number of significant prototypes per class, shape (num_classes,).
        """
        weights = torch.relu(self.classifier.weight.detach())
        return (weights > threshold).sum(dim=1)
    
    def get_prototype_importance(self) -> torch.Tensor:
        """Calculate average weight per prototype across all classes.
        
        Returns:
            torch.Tensor: Importance score per prototype, shape (num_prototypes,).
        """
        return torch.relu(self.classifier.weight.detach()).mean(dim=0)
    
    def limit_prototypes(self, k: int) -> int:
        """Limit classifier to use only top-k prototypes per class.
        
        Masks classifier weights to keep only the k highest-weighted prototypes
        for each class, setting all other weights to zero. This improves
        interpretability by reducing prototype usage while maintaining performance.
        
        Args:
            k (int): Number of top prototypes to keep per class.
        
        Returns:
            int: Number of unique prototypes remaining active after limiting.
        
        Raises:
            ValueError: If k is less than 1.
        """
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        
        if k > self.num_prototypes:
            logger.warning(f"k={k} > num_prototypes={self.num_prototypes}, no limiting applied")
            return self.num_prototypes
        
        with torch.no_grad():
            original_weights = self.classifier.weight.data.clone()
            positive_weights = torch.relu(original_weights)
            
            # Get top-k indices per class
            _, top_k_indices = torch.topk(positive_weights, k=k, dim=1)
            
            # Create binary mask
            mask = torch.zeros_like(original_weights)
            mask.scatter_(1, top_k_indices, 1.0)
            
            # Apply mask to weights
            self.classifier.weight.data = original_weights * mask
            
            # Count unique active prototypes
            unique_active = int(mask.any(dim=0).sum().item())
            logger.info(f"Limited to top-{k} prototypes per class: "
                       f"{unique_active}/{self.num_prototypes} prototypes remain active")
            
            return unique_active
