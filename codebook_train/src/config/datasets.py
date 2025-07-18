from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class BaseDatasetConfig:
    name: str = MISSING
    _path: str = MISSING
    num_classes: int = MISSING


@dataclass
class CUB200Config(BaseDatasetConfig):
    name: str = "cub200"
    num_classes: int = 200
    resize_size: int = 256
    crop_size: int = 224
    random_erase: Optional[float] = 0.1
    horizontal_flip: Optional[float] = 0.5


@dataclass
class ImageNet1KConfig(BaseDatasetConfig):
    name: str = "imagenet1k"
    num_classes: int = 1000
    resize_size: int = 256
    crop_size: int = 224
    random_erase: Optional[float] = 0.1
    horizontal_flip: Optional[float] = 0.5


@dataclass
class StanfordCarsConfig(BaseDatasetConfig):
    name: str = "stanford_cars"
    num_classes: int = 196
    resize_size: int = 256
    crop_size: int = 224
    random_erase: Optional[float] = 0.1
    horizontal_flip: Optional[float] = 0.5


@dataclass
class Flowers102Config(BaseDatasetConfig):
    name: str = "flowers102"
    num_classes: int = 102
    resize_size: int = 256
    crop_size: int = 224
    random_erase: Optional[float] = 0.1
    horizontal_flip: Optional[float] = 0.5


config_store = ConfigStore.instance()
config_store.store(group="dataset", name="cub200", node=CUB200Config)
config_store.store(group="dataset", name="imagenet1k", node=ImageNet1KConfig)
config_store.store(group="dataset", name="stanford_cars", node=StanfordCarsConfig)
config_store.store(group="dataset", name="flowers102", node=Flowers102Config)
