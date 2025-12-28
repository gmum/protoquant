from src.config.datasets import BaseDatasetConfig
from src.config.main_config import DataloaderConfig, MainConfig
from src.datasets.imagenet import get_imagenet1k
from src.datasets.cub import get_cub200
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from src.datasets.stanford_cars import get_stanford_cars
from src.datasets.flowers102 import get_flowers102
from src.datasets.funnybirds import FunnyBirdsClassification, get_funnybirds

from torchvision.datasets import StanfordCars, Flowers102, ImageNet
from src.datasets.cub import CUB200
from src.datasets.stanford_dogs import StanfordDogs, get_stanford_dogs
from torchvision.transforms.v2 import Compose

AVAILABLE_DATASET = CUB200 | StanfordCars | Flowers102 | ImageNet | StanfordDogs | FunnyBirdsClassification

DATASET_DICT = {
    "cub200": get_cub200,
    "imagenet1k": get_imagenet1k,
    "stanford_cars": get_stanford_cars,
    "flowers102": get_flowers102,
    "stanford_dogs": get_stanford_dogs,
    "funnybirds": get_funnybirds,
}

def get_dataset(
    name: str,
    path: str, 
    train_transform: Compose, 
    val_transform: Compose,
) -> tuple[AVAILABLE_DATASET, AVAILABLE_DATASET]:
    """Constructs the training and validation datasets based on the provided configuration.

    Args:
        name (str): Name of the dataset to be used.
        path (str): Path to the dataset.
        train_transform (Compose): Transformations to be applied to the training dataset.
        val_transform (Compose): Transformations to be applied to the validation dataset.

    Raises:
        ValueError: If the specified dataset name is not supported.

    Returns:
        tuple[AVAILABLE_DATASET, AVAILABLE_DATASET]: Returns the training and validation datasets.
    """
    if name not in DATASET_DICT:
        raise ValueError(f"Dataset {name} is not supported. Available datasets: {list(DATASET_DICT.keys())}")

    train_dataset, val_dataset = DATASET_DICT[name](path, train_transform, val_transform)

    return train_dataset, val_dataset


def get_dataloaders(
    train_dl_config: DataloaderConfig,
    val_dl_config: DataloaderConfig,
    train_dataset: Dataset[AVAILABLE_DATASET],
    val_dataset: Dataset[AVAILABLE_DATASET],
    train_sampler: DistributedSampler[AVAILABLE_DATASET] | None = None,
    val_sampler: DistributedSampler[AVAILABLE_DATASET] | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Constructs the dataloaders for training and validation datasets.

    Args:
        train_dl_config (DataloaderConfig): Configuration for the training dataloader.
        val_dl_config (DataloaderConfig): Configuration for the validation dataloader.
        train_dataset (Dataset[AVAILABLE_DATASET]): The training dataset.
        val_dataset (Dataset[AVAILABLE_DATASET]): The validation dataset.
        train_sampler (DistributedSampler[AVAILABLE_DATASET] | None, optional): Sampler for the training dataset. Defaults to None.
        val_sampler (DistributedSampler[AVAILABLE_DATASET] | None, optional): Sampler for the validation dataset. Defaults to None.

    Returns:
        tuple[DataLoader, DataLoader]: Returns the training and validation dataloaders.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_dl_config.batch_size,
        shuffle=None if train_sampler else True,
        pin_memory=train_dl_config.pin_memory,
        num_workers=train_dl_config.num_workers,
        drop_last=train_dl_config.drop_last,
        persistent_workers=True,
        sampler=train_sampler,
    )

    validation_loader = DataLoader(
        val_dataset,
        batch_size=val_dl_config.batch_size,
        shuffle=None if val_sampler else False,
        pin_memory=val_dl_config.pin_memory,
        num_workers=val_dl_config.num_workers,
        drop_last=val_dl_config.drop_last,
        persistent_workers=True,
        sampler=val_sampler,
    )

    return train_loader, validation_loader
