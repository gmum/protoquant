from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class DataloaderConfig:
    batch_size: int = 32
    num_workers: int = 8
    pin_memory: bool = True
    drop_last: bool = True


@dataclass
class BaseDatasetConfig:
    name: str = MISSING
    _path: str = MISSING
    num_classes: int = MISSING

    # default ones (imagenet)
    resize_size: int | None = 256
    crop_size: int | None = 224  # if precropped set to None
    random_erase: float | None = 0.1
    horizontal_flip: float | None = 0.5
    is_precropped: bool = False
    autoaugment: bool = False

    use_deit_transforms: bool = False

    # Resize to 224 and normalize (no augmentations)
    use_resize_norm_transforms: bool = False

    # Some benchmarks (e.g. FunnyBirds) use raw tensors without ImageNet normalization
    # and without additional augmentation. When enabled, training code should use
    # identity transforms.
    use_raw_transforms: bool = False


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


@dataclass
class FunnyBirdsConfig(BaseDatasetConfig):
    name: str = "funnybirds"
    num_classes: int = 50

    # FunnyBirds images are already 256x256 PNGs; official framework applies no transforms
    resize_size: int | None = None
    crop_size: int | None = None
    random_erase: float | None = None
    horizontal_flip: float | None = None
    autoaugment: bool = False
    use_deit_transforms: bool = False
    use_raw_transforms: bool = True


config_store = ConfigStore.instance()
config_store.store(group="dataset", name="cub200", node=CUB200Config)
config_store.store(group="dataset", name="imagenet1k", node=ImageNet1KConfig)
config_store.store(group="dataset", name="stanford_cars", node=StanfordCarsConfig)
config_store.store(group="dataset", name="flowers102", node=Flowers102Config)
config_store.store(group="dataset", name="stanford_dogs", node=StanfordDogsConfig)
config_store.store(group="dataset", name="funnybirds", node=FunnyBirdsConfig)
