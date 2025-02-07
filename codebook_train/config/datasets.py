from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class BaseDataset:
    name: str = MISSING
    _path: str = MISSING


@dataclass
class CUB200(BaseDataset):
    name: str = "cub200"
    num_classes: int = 200
    resize_size: int = 256
    crop_size: int = 224
    random_erase: Optional[float] = 0.1
    horizontal_flip: Optional[float] = 0.5


@dataclass
class ImageNet1K(BaseDataset):
    name: str = "imagenet1k"
    num_classes: int = 1000
    resize_size: int = 256
    crop_size: int = 224
    random_crop: Optional[float] = 0.1
    horizontal_flip: Optional[float] = 0.5


config_store = ConfigStore.instance()
config_store.store(group="dataset", name="cub200", node=CUB200)
config_store.store(group="dataset", name="imagenet1k", node=ImageNet1K)
