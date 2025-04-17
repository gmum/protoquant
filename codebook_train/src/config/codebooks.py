from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from src.codebook import (
    CosineSimilarityCodebook,
    DimReductionWrapper,
    VectorQuantizeCodebook,
)

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


@dataclass
class VectorQuantizeCodebookConfig(BaseCodebookConfig):
    _target_: str = (
        f"{VectorQuantizeCodebook.__module__}.{VectorQuantizeCodebook.__qualname__}"
    )
    # https://github.com/lucidrains/vector-quantize-pytorch/blob/8bb740e8dfcb6cd2f76648f971a211dd0d8da766/vector_quantize_pytorch/vector_quantize_pytorch.py#L846

    learnable_codebook: bool = True
    ema_update: bool = False
    layernorm_after_project_in: bool = True

    # general
    decay: float = 0.8
    eps: float = 1e-5
    use_cosine_sim: bool = False

    # initialization
    heads: int = 1
    kmeans_init: bool = False
    kmeans_iters: int = 10

    # losses
    commitment_weight: float = 1.0
    orthogonal_reg_weight: float = 0.0
    codebook_diversity_loss_weight: float = 0.0


config_store = ConfigStore.instance()
config_store.store(group="codebook", name="cosine", node=CosineSimilarityCodebookConfig)
config_store.store(
    group="codebook", name="dim_reduction", node=DimReductionWrapperConfig
)
config_store.store(
    group="codebook", name="vector_quantize", node=VectorQuantizeCodebookConfig
)
