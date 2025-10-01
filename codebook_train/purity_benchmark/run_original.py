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

# --- Import your defined config and dataset function ---
from purity_benchmark.original_config import OfficialBenchConfig
from src.datasets.construct_dataset import get_dataset

# --- Import the working benchmark logic from your main script ---
# We assume your main script is named 'start_purity.py' and is in the same directory.
from .start_purity import generate_purity_csv, eval_purity_from_csv

# --- PIPNet-provided code for their model architecture ---
from .features.convnext_features import convnext_tiny_13_features

@dataclass
class PIPNetOutput:
    proto_fmap: torch.Tensor
    proto_fvec: torch.Tensor
    logits: torch.Tensor

class OfficialPIPNet(nn.Module):
    def __init__(self, num_classes: int, num_prototypes: int, feature_net: nn.Module, add_on_layers: nn.Module, pool_layer: nn.Module, classification_layer: nn.Module):
        super().__init__()
        self._num_classes, self._num_prototypes = num_classes, num_prototypes
        self._net, self._add_on, self._pool, self._classification = feature_net, add_on_layers, pool_layer, classification_layer
    def forward(self, xs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._net(xs)
        proto_features = self._add_on(features)
        pooled = self._pool(proto_features)
        out = self._classification(pooled)
        return proto_features, pooled, out

class NonNegLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.normalization_multiplier = nn.Parameter(torch.ones((1,), requires_grad=True))
        if bias: self.bias = nn.Parameter(torch.empty(out_features))
        else: self.register_parameter('bias', None)
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return F.linear(input_tensor, torch.relu(self.weight), self.bias)

def load_official_pipnet_model(num_classes: int, checkpoint_path: str) -> OfficialPIPNet:
    """Builds the official PIPNet architecture and loads the state_dict."""
    args = Namespace(net='convnext_tiny_13', num_features=0, bias=False, disable_pretrained=True)
    features = convnext_tiny_13_features(pretrained=not args.disable_pretrained)
    first_add_on_layer_in_channels = [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
    num_prototypes = first_add_on_layer_in_channels
    add_on_layers = nn.Sequential(nn.Softmax(dim=1))
    pool_layer = nn.Sequential(nn.AdaptiveMaxPool2d(output_size=(1, 1)), nn.Flatten())
    classification_layer = NonNegLinear(num_prototypes, num_classes, bias=args.bias)
    model = OfficialPIPNet(num_classes, num_prototypes, features, add_on_layers, pool_layer, classification_layer)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # --- FIX IS HERE: Unwrap the state_dict ---
    # Create a new state_dict without the 'module.' prefix
    state_dict = checkpoint['model_state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # First, strip the 'module.' prefix if it exists
        name = k[7:] if k.startswith('module.') else k
        # Second, manually rename the mismatched '_multiplier' key
        if name == '_multiplier':
            name = '_classification.normalization_multiplier'
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    # --- END OF FIX ---
    
    return model

class OfficialPIPNetWrapper(nn.Module):
    """A wrapper to make the official model's output compatible with our benchmark script."""
    def __init__(self, official_model: OfficialPIPNet):
        super().__init__()
        self.model = official_model
        # Create a dummy 'head' attribute for compatibility with the imported benchmark functions
        self.head = Namespace(classifier=self.model._classification, P=self.model._num_prototypes)
    def forward(self, x: torch.Tensor) -> PIPNetOutput:
        pfs, pooled, out = self.model(x)
        return PIPNetOutput(proto_fmap=pfs, proto_fvec=pooled, logits=out)

@hydra.main(config_path=".", config_name="official_bench_config", version_base="1.2")
def run_official_benchmark(cfg: OfficialBenchConfig):
    """Main function to run the benchmark on the official PIPNet checkpoint."""
    print("--- Official PIP-Net Purity Sanity Check ---")
    print(OmegaConf.to_yaml(cfg))
    
    cfg.cub_cropped_data_path = to_absolute_path(cfg.cub_cropped_data_path)
    cfg.output_dir = to_absolute_path(cfg.output_dir)
    cfg.official_checkpoint_path = to_absolute_path(cfg.official_checkpoint_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    transform = transforms_v2.Compose([
        transforms_v2.Resize((cfg.dataset.image_size, cfg.dataset.image_size), interpolation=transforms_v2.InterpolationMode.BICUBIC, antialias=True),
        transforms_v2.ToTensor(),
        transforms_v2.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    print(f"Loading official model from: {cfg.official_checkpoint_path}")
    official_model = load_official_pipnet_model(200, cfg.official_checkpoint_path)
    model = OfficialPIPNetWrapper(official_model).to(device).eval()

    print(f"Loading dataset from: {cfg.cub_cropped_data_path}")
    _, val_ds = get_dataset(name="cub200", val_transform=transform, path=cfg.cub_cropped_data_path, train_transform=transform)
    projectloader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)

    print("\nStarting CSV Generation...")
    csv_filepath = generate_purity_csv(model, projectloader, cfg, device)
    
    print("\nStarting Evaluation...")
    eval_purity_from_csv(csv_path=csv_filepath, cfg=cfg)
    
    print("\n--- Sanity Check Finished ---")

if __name__ == "__main__":
    run_official_benchmark()
