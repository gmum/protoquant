# FILE: run_official_benchmark.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from argparse import Namespace
from typing import OrderedDict, Tuple
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from torchvision.transforms import v2 as transforms_v2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from purity_benchmark.original_config import OfficialBenchConfig
from src.datasets.construct_dataset import get_dataset
from .start_purity import generate_purity_csv, eval_purity_from_csv
from .features.convnext_features import convnext_tiny_13_features

import logging
logger = logging.getLogger(__name__)

@dataclass
class PIPNetOutput:
    proto_fmap: torch.Tensor
    proto_fvec: torch.Tensor
    logits: torch.Tensor

class OfficialPIPNet(nn.Module):
    def __init__(self, num_classes, num_prototypes, feature_net, add_on_layers, pool_layer, classification_layer):
        super().__init__()
        self._num_classes = num_classes
        self._num_prototypes = num_prototypes
        self._net = feature_net
        self._add_on = add_on_layers
        self._pool = pool_layer
        self._classification = classification_layer

    def forward(self, xs):
        features = self._net(xs)
        proto_features = self._add_on(features)
        pooled = self._pool(proto_features)
        out = self._classification(pooled)
        return proto_features, pooled, out

class NonNegLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.normalization_multiplier = nn.Parameter(torch.ones((1,), requires_grad=True))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input_tensor):
        # Important: This RELU is applied in forward, but not saved in state_dict
        weight = torch.relu(self.weight)
        multiplier = torch.relu(self.normalization_multiplier)
        return multiplier * F.linear(input_tensor, weight, self.bias)

def load_official_pipnet_model(num_classes: int, checkpoint_path: str) -> OfficialPIPNet:
    # 1. Architecture Definition matching ConvNext_tiny_13
    args = Namespace(net='convnext_tiny_13', num_features=0, bias=False, disable_pretrained=True)
    features = convnext_tiny_13_features(pretrained=False)
    
    # Get Channel count
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = features(dummy)
    # Usually 768 for ConvNext Tiny
    dim = out.shape[1] 
    
    num_prototypes = dim # Official implementation sets protos = channels
    add_on_layers = nn.Sequential(nn.Softmax(dim=1))
    pool_layer = nn.Sequential(nn.AdaptiveMaxPool2d(output_size=(1, 1)), nn.Flatten())
    classification_layer = NonNegLinear(num_prototypes, num_classes, bias=False)
    
    model = OfficialPIPNet(num_classes, num_prototypes, features, add_on_layers, pool_layer, classification_layer)

    # 2. Load State Dict
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    new_state_dict = OrderedDict()
    
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        # Mapping fix for the multiplier buffer/param
        if name == '_multiplier':
            name = '_classification.normalization_multiplier'
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    return model

class OfficialPIPNetWrapper(nn.Module):
    def __init__(self, official_model: OfficialPIPNet):
        super().__init__()
        self.model = official_model
        # Expose 'head' for the benchmark script
        self.head = Namespace(
            classifier=self.model._classification, 
            P=self.model._num_prototypes
        )
        
    def forward(self, x):
        pfs, pooled, out = self.model(x)
        return PIPNetOutput(proto_fmap=pfs, proto_fvec=pooled, logits=out)

@hydra.main(config_path=".", config_name="official_bench_config", version_base="1.2")
def run_official_benchmark(cfg: OfficialBenchConfig):
    cfg.cub_cropped_data_path = to_absolute_path(cfg.cub_cropped_data_path)
    cfg.output_dir = to_absolute_path(cfg.output_dir)
    cfg.official_checkpoint_path = to_absolute_path(cfg.official_checkpoint_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Standard Val Transform (Resize, Normalize)
    # Use this for deterministic evaluation
    eval_transform = transforms_v2.Compose([
        transforms_v2.Resize((cfg.dataset.image_size, cfg.dataset.image_size), interpolation=transforms_v2.InterpolationMode.BICUBIC, antialias=True),
        transforms_v2.ToTensor(),
        transforms_v2.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    print(f"Loading official model from: {cfg.official_checkpoint_path}")
    official_model = load_official_pipnet_model(200, cfg.official_checkpoint_path)
    model = OfficialPIPNetWrapper(official_model).to(device).eval()

    print(f"Loading TRAIN dataset (for projection/purity) from: {cfg.cub_cropped_data_path}")
    
    # --- CRITICAL FIX: Load Training Set, but apply Eval Transform ---
    # Purity is measured on the dataset used to define prototypes (Training Set)
    train_ds, _ = get_dataset(
        name="cub200", 
        path=cfg.cub_cropped_data_path, 
        train_transform=eval_transform, # Pass eval transform here to avoid random crops
        val_transform=eval_transform
    )
    
    # Use train_ds, not val_ds
    projectloader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=False)

    print("\nStarting CSV Generation...")
    # Note: We pass the projectloader (Training Data)
    csv_filepath, latent_wshape = generate_purity_csv(model, projectloader, cfg, device)
    
    print("\nStarting Evaluation...")
    eval_purity_from_csv(csv_path=csv_filepath, cfg=cfg, latent_wshape=latent_wshape)

if __name__ == "__main__":
    run_official_benchmark()