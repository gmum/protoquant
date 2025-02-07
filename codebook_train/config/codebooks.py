from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from codebook import CosineSimilarityCodebook


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


config_store = ConfigStore.instance()
config_store.store(group="codebook", name="cosine", node=CosineSimilarityCodebookConfig)
