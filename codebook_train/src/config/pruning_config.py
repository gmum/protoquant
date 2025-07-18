from dataclasses import dataclass, field
import random
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from src.config.codebooks import BaseCodebookConfig
from src.config.datasets import BaseDatasetConfig

from src.config.main_config import (
    Dataloaders,
    WandbConfig,
    ModelConfig,
    TrainingConfig,
)


@dataclass
class PruningConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"dataset": "_"},
            {"codebook": "_"},
        ]
    )
    seed: int = random.randint(0, 10**6)

    model: ModelConfig = field(default_factory=ModelConfig)

    target_num_codes: int = MISSING
    steps: int = MISSING

    dataset: BaseDatasetConfig = MISSING
    train_dataloader: Dataloaders = field(default_factory=Dataloaders)
    val_dataloader: Dataloaders = field(default_factory=Dataloaders)

    codebook: BaseCodebookConfig = MISSING
    codebook_path: str = MISSING
    training: TrainingConfig = field(default_factory=TrainingConfig)
    _logging_level: int = 20

    wandb: WandbConfig = field(default_factory=WandbConfig)
    output_checkpoint_path: Optional[str] = ""


# register the config groups
config_store = ConfigStore.instance()
config_store.store(name="pruning_config", node=PruningConfig)
