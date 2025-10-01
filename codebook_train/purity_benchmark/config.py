from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


# --- Self-Contained Configuration Schema ---
# We define minimal versions of the necessary config dataclasses directly in this script.
# This decouples the script from your project's main config files.

@dataclass
class MinimalModelConfig:
    name: str = MISSING
    checkpoint_path: Optional[str] = None # Will be ignored but needed for structure
    global_pool: str = ""

@dataclass
class MinimalDatasetConfig:
    # These values are fixed for this benchmark
    num_classes: int = 200
    image_size: int = 224

@dataclass
class MinimalTrainingConfig:
    # This must match how the model was trained
    train_codebook: bool = True

@dataclass
class PurityBenchConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
        ]
    )
    
    
    # --- Mandatory Parameters ---
    # These MUST be provided on the command line.
    checkpoint_path: str = MISSING
    cub_cropped_data_path: str = MISSING
    
    csv_to_eval: Optional[str] = None  # If provided, skip CSV generation and just evaluate this file.
    
    # --- Benchmark Settings with Sensible Defaults ---
    output_dir: str = "benchmark_results"
    k_top_patches: int = 10
    
    # --- Parameters Required by PIPNet Utilities ---
    # Default for a 224x224 input with a ResNet/ViT-style backbone.
    p_gaussian_hw: int = 28
    
    # --- Nested Configs to Reconstruct the Model ---
    model: MinimalModelConfig = field(default_factory=MinimalModelConfig)
    dataset: MinimalDatasetConfig = field(default_factory=MinimalDatasetConfig)
    training: MinimalTrainingConfig = field(default_factory=MinimalTrainingConfig)
    
    # Internal fields used by the model builder function
    pipnet_checkpoint_path: Optional[str] = None
    codebook_path: Optional[str] = None


# Register the complete configuration schema with Hydra's ConfigStore.
# This allows Hydra to build the config object without a YAML file.
cs = ConfigStore.instance()
cs.store(name="purity_config", node=PurityBenchConfig)
