from dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class DistributedConfig:
    """
    Configuration for distributed training.
    """

    backend: str = "nccl"
    init_method: str = MISSING
    world_size: int = MISSING
