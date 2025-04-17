from dataclasses import dataclass, field
import random
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from src.config.codebooks import BaseCodebookConfig
from src.config.datasets import BaseDataset
from src.config.optimizers import BaseOptimizerConfig


@dataclass
class Dataloaders:
    batch_size: int = 32
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
class TrainingConfig:
    unfreeze_before: int = 0  # how many layers to unfreeze before the codebook
    label_smoothing: float = 0.1
    enable_schedulers: bool = True
    warmup_epochs: int = MISSING
    restart_threshold: int = -1

    task_loss_weight: float = 1.0
    codebook_loss_weight: float = 1.0

@dataclass
class MainConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"base_optimizer": "none"},
            {"codebook_optimizer": "_"},
            {"dataset": "_"},
            {"codebook": "_"},
        ]
    )
    seed: int = random.randint(0, 1e6)

    model: ModelConfig = field(default_factory=ModelConfig)
    base_optimizer: BaseOptimizerConfig = MISSING
    codebook_optimizer: BaseOptimizerConfig = MISSING

    epochs: int = MISSING

    dataset: BaseDataset = MISSING
    train_dataloader: Dataloaders = field(default_factory=Dataloaders)
    val_dataloader: Dataloaders = field(default_factory=Dataloaders)

    codebook: BaseCodebookConfig = MISSING
    codebook_path: Optional[str] = None
    training: TrainingConfig = field(default_factory=TrainingConfig)
    _logging_level: int = 20

    wandb: WandbConfig = field(default_factory=WandbConfig)
    output_checkpoint_path: Optional[str] = ""


# register the config groups
config_store = ConfigStore.instance()
config_store.store(name="main_config", node=MainConfig)
