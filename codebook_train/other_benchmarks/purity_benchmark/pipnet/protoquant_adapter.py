"""Benchmark-local adapter to make ProtoQuantNet present a PIPNet-like API.

The adapter exposes the following PIPNet-like attributes/methods used by the
purity benchmark code:
 - forward(x) -> (softmax_map, pooled_proto_scores, logits)
 - _num_prototypes (int)
 - _num_classes (int)
 - _classification (nn.Module with .weight and .bias referencing underlying classifier)
 - _multiplier (nn.Parameter)
 - calculate_local_size()
 - get_prototype_importance()

This adapter intentionally keeps things minimal and benchmark-local.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ClassifierShim(nn.Module):
    """Small shim to expose .weight and .bias attributes expected by benchmark code."""

    def __init__(self, classifier: nn.Module):
        super().__init__()
        # Reuse the underlying parameters so that updates / state_dicts remain consistent.
        # Assigning Parameter objects registers them on this module as well.
        self.weight = classifier.weight
        self.bias = getattr(classifier, "bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Provide a forward pass fallback in case something calls it directly.
        if self.bias is None:
            return F.linear(x, F.relu(self.weight))
        else:
            return F.linear(x, F.relu(self.weight), self.bias)


class ProtoQuantAdapter(nn.Module):
    """Adapter that wraps a ProtoQuantNet and exposes a PIPNet-compatible API.

    Notes:
    - `forward(x)` returns (softmax_map, pooled_proto_scores, logits) similar to PIPNet.
    - `softmax_map` is computed as softmax(similarity_map / temperature, dim=1).
    - `_classification.weight` refers to the underlying classifier weights (Parameter).
    - `_multiplier` is provided as a Parameter for code that expects it; default value 1.0.
    """

    def __init__(self, protoquant_model: "ProtoQuantNet") -> None:
        super().__init__()
        self.model = protoquant_model

        # Keep names similar to PIPNet expectations
        self._num_prototypes = int(getattr(self.model, "num_prototypes"))
        self._num_classes = int(getattr(self.model, "num_classes"))

        # Expose a classification shim with .weight and .bias
        self._classification = _ClassifierShim(self.model.classifier)

        # _multiplier is used by benchmark code (created as a Parameter so code can set requires_grad)
        self._multiplier = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (softmax_map, pooled_proto_scores, logits).

        softmax_map: (B, P, H, W)
        pooled_proto_scores: (B, P)
        logits: (B, num_classes)
        """
        out = self.model(x, return_similarity_map=True)

        similarity_map = out.similarity_map
        if similarity_map is None:
            raise RuntimeError("ProtoQuantNet returned no similarity_map; ensure return_similarity_map=True")

        # Softmax over prototype/channel dimension to mimic PIPNet semantics
        softmax_map = F.softmax(similarity_map / float(self.model.temperature), dim=1)

        pooled = out.pooled_proto_scores
        logits = out.logits if out.logits is not None else self._classification(pooled)

        return softmax_map, pooled, logits

    # Delegate convenience methods
    def calculate_local_size(self, threshold: float = 0.1) -> torch.Tensor:
        return self.model.calculate_local_size(threshold)

    def get_prototype_importance(self) -> torch.Tensor:
        return self.model.get_prototype_importance()


# Lightweight import guard for type checking / IDEs
try:
    # Import type only for hints; if src is unavailable this will not raise at runtime
    from src.models.proto_quantnet import ProtoQuantNet  # type: ignore
except Exception:
    ProtoQuantNet = Optional[object]  # fallback for type checkers
