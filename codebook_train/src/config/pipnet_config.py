from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from src.config.distributed import DistributedConfig
from src.config.datasets import BaseDatasetConfig, DataloaderConfig
from src.config.optimizers import BaseOptimizerConfig
from logging import INFO


@dataclass
class WandbConfig:
    is_enabled: bool = False
    project: str = MISSING
    entity: Optional[str] = None
    group: Optional[str] = None
    job_type: Optional[str] = None
    tags: list[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    name: str = MISSING
    checkpoint_path: Optional[str] = None
    global_pool: Optional[str] = "avg" # for vit


@dataclass
class TrainingConfig:
    unfreeze_before: int = 0  # how many layers to unfreeze before the codebook
    label_smoothing: float = 0.1
    enable_schedulers: bool = True
    warmup_epochs: int = MISSING
    train_codebook: bool = True
    classifier_sparsity_lambda: float = 0.0
    use_random_codes: bool = False  # Replace loaded codes with random normal distribution


@dataclass
class PipNetConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"base_optimizer": "none"},
            {"dataset": "_"},
        ]
    )
    seed: int = MISSING

    model: ModelConfig = field(default_factory=ModelConfig)
    base_optimizer: BaseOptimizerConfig = MISSING
    codebook_path: Optional[str] = None  # Path to codebook tensor state_dict
    pipnet_checkpoint_path: Optional[str] = None  # Path to a full PIPNet model checkpoint

    epochs: int = MISSING
    dataset: BaseDatasetConfig = MISSING
    train_dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    val_dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)

    training: TrainingConfig = field(default_factory=TrainingConfig)
    _logging_level: int = INFO

    wandb: WandbConfig = field(default_factory=WandbConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)


# register the config groups
config_store = ConfigStore.instance()
config_store.store(name="pipnet_config", node=PipNetConfig)
