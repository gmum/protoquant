# FILE: purity_benchmark/original_config.py

from dataclasses import dataclass, field
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore


# Define minimal dataclasses that match the structure your benchmark functions expect
@dataclass
class ModelConfig:
    name: str = "convnext_tiny_13"  # Hardcoded, as this script is only for this model


@dataclass
class DatasetConfig:
    image_size: int = 224


# The main config for the official benchmark script
@dataclass
class OfficialBenchConfig:
    official_checkpoint_path: str = MISSING
    cub_cropped_data_path: str = MISSING
    output_dir: str = "official_benchmark_results"
    k_top_patches: int = 10

    # These fields are required for compatibility with the imported functions
    p_gaussian_hw: int = 28  # Value for ConvNeXt
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)


# Register the schema with Hydra
cs = ConfigStore.instance()
cs.store(name="official_bench_config", node=OfficialBenchConfig)
