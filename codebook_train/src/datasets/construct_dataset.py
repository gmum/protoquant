from src.config.main_config import DataloaderConfig, MainConfig
from src.datasets.imagenet import get_imagenet1k
from src.datasets.cub import get_cub200
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from src.datasets.stanford_cars import get_stanford_cars
from src.datasets.flowers102 import get_flowers102

from torchvision.datasets import StanfordCars, Flowers102, ImageNet
from src.datasets.cub import CUB200
from src.datasets.stanford_dogs import StanfordDogs, get_stanford_dogs

AVAILABLE_DATASET = CUB200 | StanfordCars | Flowers102 | ImageNet | StanfordDogs


def get_dataset(
    cfg: MainConfig,
) -> tuple[AVAILABLE_DATASET, AVAILABLE_DATASET]:
    if cfg.dataset.name == "imagenet1k":
        return get_imagenet1k(
            path=cfg.dataset._path,
            crop_value=cfg.dataset.crop_size,
            resize_value=cfg.dataset.resize_size,
            random_erase=cfg.dataset.random_erase,
            horizontal_flip=cfg.dataset.horizontal_flip,
            is_precropped=cfg.dataset.is_precropped,
        )
    elif cfg.dataset.name == "cub200":
        return get_cub200(
            path=cfg.dataset._path,
            crop_value=cfg.dataset.crop_size,
            resize_value=cfg.dataset.resize_size,
            random_erase=cfg.dataset.random_erase,
            horizontal_flip=cfg.dataset.horizontal_flip,
            is_precropped=cfg.dataset.is_precropped,
        )
    elif cfg.dataset.name == "stanford_cars":
        return get_stanford_cars(
            path=cfg.dataset._path,
            crop_value=cfg.dataset.crop_size,
            resize_value=cfg.dataset.resize_size,
            random_erase=cfg.dataset.random_erase,
            horizontal_flip=cfg.dataset.horizontal_flip,
            is_precropped=cfg.dataset.is_precropped,
        )
    elif cfg.dataset.name == "flowers102":
        return get_flowers102(
            path=cfg.dataset._path,
            crop_value=cfg.dataset.crop_size,
            resize_value=cfg.dataset.resize_size,
            random_erase=cfg.dataset.random_erase,
            horizontal_flip=cfg.dataset.horizontal_flip,
            is_precropped=cfg.dataset.is_precropped,
        )
    elif cfg.dataset.name == "stanford_dogs":
        return get_stanford_dogs(
            path=cfg.dataset._path,
            crop_value=cfg.dataset.crop_size,
            resize_value=cfg.dataset.resize_size,
            random_erase=cfg.dataset.random_erase,
            horizontal_flip=cfg.dataset.horizontal_flip,
            is_precropped=cfg.dataset.is_precropped,
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")


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
