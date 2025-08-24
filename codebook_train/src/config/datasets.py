from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class BaseDatasetConfig:
    name: str = MISSING
    _path: str = MISSING
    num_classes: int = MISSING
    
    # default ones (imagenet)
    resize_size: int | None = 256
    crop_size: int | None = 224 # if precropped set to None
    random_erase: float | None = 0.1
    horizontal_flip: float | None = 0.5
    is_precropped: bool = False


@dataclass
class CUB200Config(BaseDatasetConfig):
    name: str = "cub200"
    num_classes: int = 200
    resize_size: int = 224
    crop_size: int | None = None
    is_precropped: bool = True


@dataclass
class ImageNet1KConfig(BaseDatasetConfig):
    name: str = "imagenet1k"
    num_classes: int = 1000


@dataclass
class StanfordCarsConfig(BaseDatasetConfig):
    name: str = "stanford_cars"
    num_classes: int = 196


@dataclass
class Flowers102Config(BaseDatasetConfig):
    name: str = "flowers102"
    num_classes: int = 102


@dataclass
class StanfordDogsConfig(BaseDatasetConfig):
    name: str = "stanford_dogs"
    num_classes: int = 120


config_store = ConfigStore.instance()
config_store.store(group="dataset", name="cub200", node=CUB200Config)
config_store.store(group="dataset", name="imagenet1k", node=ImageNet1KConfig)
config_store.store(group="dataset", name="stanford_cars", node=StanfordCarsConfig)
config_store.store(group="dataset", name="flowers102", node=Flowers102Config)
config_store.store(group="dataset", name="stanford_dogs", node=StanfordDogsConfig)
