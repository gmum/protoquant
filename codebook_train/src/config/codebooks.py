from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from src.codebook import CosineSimilarityCodebook, DimReductionWrapper


@dataclass
class BaseCodebookConfig:
    _target_: str = MISSING
    embedding_dim: int = MISSING
    num_entries: int = MISSING


@dataclass
class CosineSimilarityCodebookConfig(BaseCodebookConfig):
    _target_: str = (
        f"{CosineSimilarityCodebook.__module__}.{CosineSimilarityCodebook.__qualname__}"
    )
    mapping_dim_config: list[int] = field(default_factory=list)


@dataclass
class DimReductionWrapperConfig(BaseCodebookConfig):
    _target_: str = (
        f"{DimReductionWrapper.__module__}.{DimReductionWrapper.__qualname__}"
    )
    input_dim: int = MISSING
    in_block_config: list[int] = MISSING
    out_block_config: list[int] = MISSING
    mapping_dim_config: list[int] = field(default_factory=list)

config_store = ConfigStore.instance()
config_store.store(group="codebook", name="cosine", node=CosineSimilarityCodebookConfig)
config_store.store(
    group="codebook", name="dim_reduction", node=DimReductionWrapperConfig
)
