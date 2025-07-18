from src.config.main_config import MainConfig
from src.datasets.imagenet import get_imagenet1k
from src.datasets.cub import get_cub200
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from src.datasets.stanford_cars import get_stanford_cars
from src.datasets.flowers102 import get_flowers102
from src.config.datasets import (
    CUB200Config,
    ImageNet1KConfig,
    StanfordCarsConfig,
    Flowers102Config,
)
import torch
from torchvision.transforms import v2 as transforms_v2
from src.config.constants import IMAGENET1K_MEAN, IMAGENET1K_STD
from torchvision.datasets import CUB200, StanfordCars, Flowers102, ImageNet
from src.datasets.cub import CUB200

AVAILABLE_DATASET = CUB200 | StanfordCars | Flowers102 | ImageNet

def get_dataset(
    cfg: MainConfig,
) -> tuple[AVAILABLE_DATASET, AVAILABLE_DATASET]:
    if isinstance(cfg.dataset, ImageNet1KConfig):
        return get_imagenet1k(
            path=cfg.dataset._path,
            crop_value=cfg.dataset.crop_size,
            resize_value=cfg.dataset.resize_size,
            random_erase=cfg.dataset.random_erase,
            horizontal_flip=cfg.dataset.horizontal_flip,
        )
    elif isinstance(cfg.dataset, CUB200Config):
        return get_cub200(
            path=cfg.dataset._path,
            crop_value=cfg.dataset.crop_size,
            resize_value=cfg.dataset.resize_size,
            random_erase=cfg.dataset.random_erase,
            horizontal_flip=cfg.dataset.horizontal_flip,
        )
    elif isinstance(cfg.dataset, StanfordCarsConfig):
        return get_stanford_cars(
            path=cfg.dataset._path,
            crop_value=cfg.dataset.crop_size,
            resize_value=cfg.dataset.resize_size,
            random_erase=cfg.dataset.random_erase,
            horizontal_flip=cfg.dataset.horizontal_flip,
        )
    elif isinstance(cfg.dataset, Flowers102Config):
        return get_flowers102(
            path=cfg.dataset._path,
            crop_value=cfg.dataset.crop_size,
            resize_value=cfg.dataset.resize_size,
            random_erase=cfg.dataset.random_erase,
            horizontal_flip=cfg.dataset.horizontal_flip,
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")


def get_dataloaders(
    cfg: MainConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    train_sampler: DistributedSampler | None = None,
    val_sampler: DistributedSampler | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Constructs the dataloaders for training and validation datasets.

    Args:
        cfg (MainConfig):  Main configuration object
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset):  The validation dataset.
        train_sampler (DistributedSampler | None, optional): Training sampler for distributed training. Defaults to None.
        val_sampler (DistributedSampler | None, optional): Validation sampler for distributed training. Defaults to None.

    Returns:
        tuple[DataLoader, DataLoader]: _description_
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_dataloader.batch_size,
        shuffle=None if train_sampler else True,
        pin_memory=cfg.train_dataloader.pin_memory,
        num_workers=cfg.train_dataloader.num_workers,
        drop_last=cfg.train_dataloader.drop_last,
        persistent_workers=True,
        sampler=train_sampler,
    )

    validation_loader = DataLoader(
        val_dataset,
        batch_size=cfg.val_dataloader.batch_size,
        shuffle=None if val_sampler else False,
        pin_memory=cfg.val_dataloader.pin_memory,
        num_workers=cfg.val_dataloader.num_workers,
        drop_last=cfg.val_dataloader.drop_last,
        persistent_workers=True,
        sampler=val_sampler,
    )

    return train_loader, validation_loader


def get_default_image_transforms(
    resize_value: int | None = None,
    crop_value: int | None = None,
    random_erase: float | None = None,
    horizontal_flip: float | None = None,
) -> tuple[transforms_v2.Compose, transforms_v2.Compose]:
    """Constructs a default set of image transforms for training and validation.

    This is a generic function suitable for many image classification tasks.

    Args:
        resize_value (int | None): The size to resize the images to. Defaults to None.
        crop_value (int | None): The size to crop the images to. Defaults to None.
        random_erase (float | None): The probability of applying random erasing. Defaults to None.
        horizontal_flip (float | None): The probability of applying horizontal flip. Defaults to None.

    Returns:
        tuple[transforms_v2.Compose, transforms_v2.Compose]: Train and validation transforms.
    """
    train_transforms = []
    val_transforms = []

    if resize_value is not None:
        train_transforms.append(transforms_v2.Resize(size=resize_value, antialias=True))
        val_transforms.append(transforms_v2.Resize(size=resize_value, antialias=True))

    if horizontal_flip is not None:
        train_transforms.append(transforms_v2.RandomHorizontalFlip(p=horizontal_flip))

    if crop_value is not None:
        train_transforms.append(transforms_v2.RandomCrop(size=crop_value))
        val_transforms.append(transforms_v2.CenterCrop(crop_value))

    train_transforms.append(transforms_v2.TrivialAugmentWide())

    common = [
        transforms_v2.ToImage(),
        transforms_v2.ToDtype(torch.float32, scale=True),
        transforms_v2.Normalize(mean=IMAGENET1K_MEAN, std=IMAGENET1K_STD),
    ]

    train_transforms.extend(common)
    val_transforms.extend(common)

    if random_erase is not None:
        train_transforms.append(transforms_v2.RandomErasing(p=random_erase))

    train_transform = transforms_v2.Compose(train_transforms)
    val_transform = transforms_v2.Compose(val_transforms)

    return train_transform, val_transform
