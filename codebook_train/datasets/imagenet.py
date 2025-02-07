import torch

from torchvision.transforms import v2 as transforms_v2

# import dataloaders
from torchvision import datasets
from torch.utils.data import Dataset
from config.constants import IMAGENET1K_MEAN, IMAGENET1K_STD


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

    common_transformations = []

    if resize_value is not None:
        common_transformations.append(
            transforms_v2.Resize((resize_value, resize_value), antialias=True)
        )

    if crop_value is not None:
        common_transformations.append(
            transforms_v2.RandomResizedCrop(crop_value, antialias=True)
        )

    common_transformations.extend(
        [
            transforms_v2.ToImage(),
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Normalize(mean=IMAGENET1K_MEAN, std=IMAGENET1K_STD),
        ]
    )

    train_transforms = common_transformations.copy()
    if horizontal_flip is not None:
        train_transforms.insert(-2, transforms_v2.RandomHorizontalFlip(horizontal_flip))

    if random_erase is not None:
        train_transforms.insert(-2, transforms_v2.RandomErasing(random_erase))

    test_transform = transforms_v2.Compose(common_transformations)
    train_transform = transforms_v2.Compose(train_transforms)

    return train_transform, test_transform
