from dataclasses import dataclass, field
import random
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from .codebooks import BaseCodebookConfig
from .datasets import BaseDataset
from .optimizers import BaseOptimizerConfig


@dataclass
class Dataloaders:
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False


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
class MainConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"optimizer": "_"},
            {"dataset": "_"},
            {"codebook": "_"},
        ]
    )
    seed: int = random.randint(0, 1e6)

    model: ModelConfig = field(default_factory=ModelConfig)
    epochs: int = MISSING
    optimizer: BaseOptimizerConfig = MISSING

    dataset: BaseDataset = MISSING
    train_dataloader: Dataloaders = field(default_factory=Dataloaders)
    val_dataloader: Dataloaders = field(default_factory=Dataloaders)

    codebook: BaseCodebookConfig = MISSING
    _logging_level: int = 20

    wandb: WandbConfig = field(default_factory=WandbConfig)
    output_checkpoint_path: Optional[str] = ""


# register the config groups
config_store = ConfigStore.instance()
config_store.store(name="main_config", node=MainConfig)
