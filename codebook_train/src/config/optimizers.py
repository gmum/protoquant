from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torch.optim import SGD, Adam, AdamW


@dataclass
class BaseOptimizerConfig:
    _target_: str = MISSING
    lr: float = MISSING


@dataclass
class AdamOptimizerConfig(BaseOptimizerConfig):
    _target_: str = f"{Adam.__module__}.{Adam.__qualname__}"
    weight_decay: float = 0.0


@dataclass
class AdamWOptimizerConfig(BaseOptimizerConfig):
    _target_: str = f"{AdamW.__module__}.{AdamW.__qualname__}"
    weight_decay: float = 0.005
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class SGDOptimizerConfig(BaseOptimizerConfig):
    _target_: str = f"{SGD.__module__}.{SGD.__qualname__}"
    momentum: float = 0.9
    weight_decay: float = 1e-4
    nesterov: bool = True


config_store = ConfigStore.instance()
for group in ("base_optimizer", "codebook_optimizer", "probe_optimizer"):
    config_store.store(group=group, name="adam", node=AdamOptimizerConfig)
    config_store.store(group=group, name="adamw", node=AdamWOptimizerConfig)
    config_store.store(group=group, name="sgd", node=SGDOptimizerConfig)

# placeholder for no optimizer
config_store.store(
    group="base_optimizer",
    name="none",
    node=BaseOptimizerConfig(
        lr=0.1,
        _target_="[Warning] Base optimizer not set and layers outside the codebook require gradients",
    ),
)
