from torch.nn.init import (
    uniform_,
    normal_,
    orthogonal_,
    xavier_normal_,
    xavier_uniform_,
    constant_,
    sparse_,
)
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class BaseInitializationConfig:
    _target_: str = MISSING


@dataclass
class UniformInitializationConfig(BaseInitializationConfig):
    _target_: str = f"{uniform_.__module__}.{uniform_.__qualname__}"
    a: float = 0.0
    b: float = 1.0


@dataclass
class NormalInitializationConfig(BaseInitializationConfig):
    _target_: str = f"{normal_.__module__}.{normal_.__qualname__}"
    mean: float = 0.0
    std: float = 1.0


@dataclass
class OrthogonalInitializationConfig(BaseInitializationConfig):
    _target_: str = f"{orthogonal_.__module__}.{orthogonal_.__qualname__}"
    gain: float = 1.0


@dataclass
class XavierNormalInitializationConfig(BaseInitializationConfig):
    _target_: str = f"{xavier_normal_.__module__}.{xavier_normal_.__qualname__}"
    gain: float = 1.0


@dataclass
class XavierUniformInitializationConfig(BaseInitializationConfig):
    _target_: str = f"{xavier_uniform_.__module__}.{xavier_uniform_.__qualname__}"
    gain: float = 1.0


@dataclass
class ConstantInitializationConfig(BaseInitializationConfig):
    _target_: str = f"{constant_.__module__}.{constant_.__qualname__}"
    val: float = 0.0


@dataclass
class SparseInitializationConfig(BaseInitializationConfig):
    _target_: str = f"{sparse_.__module__}.{sparse_.__qualname__}"
    sparsity: float = 0.1  # e.g., 10% non-zero elements
    std: float = 0.01


config_store = ConfigStore.instance()
config_store.store(
    group="codebook_init", name="uniform", node=UniformInitializationConfig
)
config_store.store(
    group="codebook_init", name="normal", node=NormalInitializationConfig
)
config_store.store(
    group="codebook_init", name="orthogonal", node=OrthogonalInitializationConfig
)
config_store.store(
    group="codebook_init", name="xavier_normal", node=XavierNormalInitializationConfig
)
config_store.store(
    group="codebook_init", name="xavier_uniform", node=XavierUniformInitializationConfig
)
config_store.store(
    group="codebook_init", name="constant", node=ConstantInitializationConfig
)
config_store.store(
    group="codebook_init", name="sparse", node=SparseInitializationConfig
)
