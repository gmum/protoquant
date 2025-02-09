import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass


@dataclass
class PIPNetOutput:
    proto_fmap: Tensor
    proto_fvec: Tensor
    logits: Tensor
    pre_softmax: Tensor
    softmax_out: Tensor


class NonNegLinear(nn.Module):
    def __init__(
        self,
        fan_in: int,
        fan_out: int,
        bias: bool = False,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((fan_out, fan_in)),
        )
        self.normalization_multiplier = nn.Parameter(
            torch.ones((1,), requires_grad=True),
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(fan_out))
        else:
            self.register_parameter('bias', None)

    def forward(self, x:  Tensor) -> Tensor:
        non_neg_weight = torch.relu(self.weight)
        return F.linear(x, non_neg_weight, self.bias)


class QuantizedPIPNetHead(nn.Module):
    def __init__(
        self, 
        num_classes: int,
        codebook: torch.Tensor, 
        normalize: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        self.K = num_classes
        self.P, self.D = codebook.shape
        self.normalize = normalize

        # Codebook as frozed conv kernel;
        self.register_buffer(
            "codebook", 
            codebook.view(self.P, self.D, 1, 1),
        )

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

    def forward(
        self, 
        x: Tensor, 
        inference: bool = False,
    ) -> PIPNetOutput:
        """
        PIP-Net-like forward 
        with frozen codebook of prototypes.
        Args:
        x: featuremap from backbone (N, C, H, W)
        inference: Inference mode.
        """
        proto_fmap = self.detect_protos(x)
        proto_fvec = self.pool(proto_fmap)
        
        if inference:
            proto_fvec = self.zero_out_irrelevant(proto_fvec)

        logits = self.classify(proto_fvec)
        pre_softmax = self.pre_softmax_norm(logits)
        softmax_out = F.softmax(pre_softmax, dim=1)

        result = PIPNetOutput(
            proto_fmap=proto_fmap,
            proto_fvec=proto_fvec,
            logits=logits,
            pre_softmax=pre_softmax,
            softmax_out=softmax_out,
        )
        return result

    def detect_protos(self, x: Tensor) -> Tensor:
        c = self.codebook
        
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
            c = F.normalize(c, p=2, dim=1)
        
        x = F.conv2d(x, c)
        x = F.softmax(x, dim=1)
        return x

    def pool(self, x: Tensor) -> Tensor:
        return self.pool_layer(x)

    def classify(self, x: Tensor) -> Tensor:
        return self.classifier(x)

    def zero_out_irrelevant(self, x: Tensor) -> Tensor:
        return torch.where(proto_fvec < 0.1, 0.0, proto_fvec)

    def pre_softmax_norm(self, x: Tensor) -> Tensor:
        return torch.log1p(x ** self.multiplier)
