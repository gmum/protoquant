from dataclasses import dataclass, field
import random
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from src.config.optimizers import BaseOptimizerConfig
from src.config.codebooks import BaseCodebookConfig
from src.config.datasets import BaseDataset

from src.config.main_config import (
    Dataloaders,
    WandbConfig,
    ModelConfig,
)


@dataclass
class SSLTrainingConfig:
    enable_schedulers: bool = True
    warmup_epochs: int = 10

    probe_epochs: int = MISSING  # number of epochs to train the linear probe
    probe_dim: int = (
        MISSING  # dimension of the linear probe, should be equal to the last layer of the model
    )

    label_smoothing: float = 0.1


@dataclass
class SelfSupervisedConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"dataset": "_"},
            {"codebook": "_"},
            {"probe_optimizer": "_"},
            {"codebook_optimizer": "_"},
        ]
    )
    seed: int = MISSING

    model: ModelConfig = field(default_factory=ModelConfig)
    epochs: int = MISSING

    dataset: BaseDataset = MISSING
    train_dataloader: Dataloaders = field(default_factory=Dataloaders)
    val_dataloader: Dataloaders = field(default_factory=Dataloaders)

    probe_optimizer: BaseOptimizerConfig = MISSING
    codebook_optimizer: BaseOptimizerConfig = MISSING

    codebook: BaseCodebookConfig = MISSING
    codebook_path: Optional[str] = None
    training: SSLTrainingConfig = field(default_factory=SSLTrainingConfig)
    _logging_level: int = 20

    wandb: WandbConfig = field(default_factory=WandbConfig)
    output_checkpoint_path: Optional[str] = ""


# register the config groups
config_store = ConfigStore.instance()
config_store.store(name="ssl_config", node=SelfSupervisedConfig)
