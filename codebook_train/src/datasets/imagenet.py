import torch

from torchvision.transforms import v2 as transforms_v2

# import dataloaders
from torchvision import datasets
from torch.utils.data import Dataset
from src.config.constants import IMAGENET1K_MEAN, IMAGENET1K_STD


def get_imagenet1k(
    path: str,
    resize_value: int | None = None,
    crop_value: int | None = None,
    random_erase: float | None = None,
    horizontal_flip: float | None = None,
) -> tuple[Dataset, Dataset]:
    """Constructs the ImageNet1K dataset.

    Args:
        path (str): Path to the dataset.
        resize_value (int | None): The size to resize the images to. Defaults to None.
        crop_value (int | None): The size to crop the images to. Defaults to None.
        random_erase (float | None): The probability of applying random erasing. Defaults to None.
        horizontal_flip (float | None): The probability of applying horizontal flip. Defaults to None.

    Returns:
        tuple[Dataset, Dataset]: Train and validation datasets.
    """

    train_transform, test_transform = get_imagenet_transforms(
        resize_value=resize_value,
        crop_value=crop_value,
        random_erase=random_erase,
        horizontal_flip=horizontal_flip,
    )

    train_dataset = datasets.ImageNet(
        root=path,
        split="train",
        transform=train_transform,
    )

    validate_dataset = datasets.ImageNet(
        root=path,
        split="val",
        transform=test_transform,
    )

    return train_dataset, validate_dataset


def get_imagenet_transforms(
    resize_value: int | None = None,
    crop_value: int | None = None,
    random_erase: float | None = None,
    horizontal_flip: float | None = None,
) -> tuple[transforms_v2.Compose, transforms_v2.Compose]:
    """Constructs the ImageNet1K dataset transforms.

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
        train_transforms.append(
            transforms_v2.Resize(size=(resize_value, resize_value), antialias=True)
        )
        val_transforms.append(
            transforms_v2.Resize(size=(resize_value, resize_value), antialias=True)
        )

    if horizontal_flip is not None:
        train_transforms.append(transforms_v2.RandomHorizontalFlip(p=horizontal_flip))

    if crop_value is not None:
        train_transforms.append(transforms_v2.RandomCrop(size=(crop_value, crop_value)))
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
