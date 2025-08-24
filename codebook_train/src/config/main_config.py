from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from src.config.distributed import DistributedConfig
from src.config.codebook_init import BaseInitializationConfig
from src.config.codebooks import BaseCodebookConfig
from src.config.datasets import BaseDatasetConfig
from src.config.optimizers import BaseOptimizerConfig
from logging import INFO


@dataclass
class DataloaderConfig:
    batch_size: int = 32
    num_workers: int = 2
    pin_memory: bool = True
    drop_last: bool = True


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
    checkpoint_path: str = MISSING


@dataclass
class TrainingConfig:
    unfreeze_before: int = 0  # how many layers to unfreeze before the codebook
    label_smoothing: float = 0.1
    enable_schedulers: bool = True
    warmup_epochs: int = MISSING
    use_amp: bool = False  # whether to use automatic mixed precision
    compile_model: bool = False
    compile_mode: str = (
        "default"  # options: "default", "reduce-overhead", "max-autotune"
    )

    task_loss_weight: float = 1.0
    codebook_loss_weight: float = 1.0

    only_features: bool = (
        False  # whether cache model features and use them to train the codebook
    )


@dataclass
class MainConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"base_optimizer": "none"},
            {"codebook_optimizer": "_"},
            {"dataset": "_"},
            {"codebook": "_"},
            {"codebook_init": "_"},
        ]
    )
    seed: int = MISSING

    model: ModelConfig = field(default_factory=ModelConfig)
    base_optimizer: BaseOptimizerConfig = MISSING
    codebook_optimizer: BaseOptimizerConfig = MISSING

    epochs: int = MISSING

    dataset: BaseDatasetConfig = MISSING
    train_dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    val_dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)

    codebook: BaseCodebookConfig = MISSING
    codebook_path: Optional[str] = None
    codebook_init: BaseInitializationConfig = MISSING
    training: TrainingConfig = field(default_factory=TrainingConfig)
    _logging_level: int = INFO

    wandb: WandbConfig = field(default_factory=WandbConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)


# register the config groups
config_store = ConfigStore.instance()
config_store.store(name="main_config", node=MainConfig)
