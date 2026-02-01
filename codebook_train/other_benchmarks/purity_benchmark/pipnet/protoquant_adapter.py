import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.proto_quantnet import ProtoQuantNet


class _NonNegLinearProxy(nn.Module):
    """A minimal NonNegLinear-compatible module.

    The purity benchmark expects:
    - .weight (Parameter)
    - .bias (Parameter or None)
    - .normalization_multiplier (Parameter)
    and uses forward(x) -> F.linear(x, relu(weight), bias)
    """

    def __init__(self, weight: nn.Parameter, bias: nn.Parameter | None):
        super().__init__()
        self.weight = weight
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = bias

        # The benchmark uses this name explicitly.
        self.normalization_multiplier = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, torch.relu(self.weight), self.bias)


class ProtoQuantAdapter(nn.Module):
    """Adapter that makes ProtoQuantNet look like PIPNet for purity benchmark code.

    It exposes the attributes used throughout the benchmark:
    - _net, _add_on, _pool, _classification, _multiplier
    - _num_classes, _num_prototypes

    And its forward returns the PIPNet tuple:
    (proto_features[B,P,H,W], pooled[B,P], out[B,C])
    """

    def __init__(self, model: ProtoQuantNet, num_classes: int):
        super().__init__()
        self._protoquant = model

        # Match ProtoQuantNet activation temperature; critical for meaningful softmax
        # when num_prototypes is large.
        self._temperature = float(getattr(model, "temperature", 0.1))

        self._num_classes = int(num_classes)
        self._num_prototypes = int(model.num_prototypes)
        self._num_features = 0

        # Provide PIPNet-like submodules/attributes.
        self._net = model.backbone
        self._add_on = nn.Identity()
        self._pool = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)), nn.Flatten())

        # Proxy classification layer with expected attribute names.
        # We use the same underlying weight/bias parameters as ProtoQuantNet.
        self._classification = _NonNegLinearProxy(
            model.classifier.weight, model.classifier.bias
        )
        self._multiplier = self._classification.normalization_multiplier

        # For benchmark compatibility: multiplier is treated as frozen in many paths.
        self._multiplier.requires_grad = False

    def forward(self, xs: torch.Tensor, inference: bool = False):
        # ProtoQuantNet can optionally return the spatial similarity map.
        out = self._protoquant(xs, return_similarity_map=True)
        if out.similarity_map is None:
            raise RuntimeError(
                "ProtoQuantNet did not return similarity_map; cannot run purity benchmark."
            )

        # Convert similarity map into PIPNet-like per-location prototype distribution.
        # IMPORTANT: match ProtoQuantNet's computation: softmax(similarity / temperature).
        # Using raw cosine similarities here would yield near-uniform maps for large P.
        proto_features = torch.softmax(out.similarity_map / self._temperature, dim=1)

        # Use the same pooling as the benchmark/PIPNet (adaptive max over spatial dims).
        # This should match ProtoQuantNet.pooled_proto_scores when computed on proto_features.
        pooled = self._pool(proto_features)
        if inference:
            pooled = torch.where(pooled < 0.1, 0.0, pooled)

        logits = self._classification(pooled)
        return proto_features, pooled, logits
