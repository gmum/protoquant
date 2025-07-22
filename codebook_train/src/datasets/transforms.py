import torch
from torchvision.transforms import v2 as transforms_v2
from src.config.constants import IMAGENET1K_MEAN, IMAGENET1K_STD

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
