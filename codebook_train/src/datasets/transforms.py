import torch
from torchvision.transforms import v2 as transforms_v2
from src.config.constants import IMAGENET1K_MEAN, IMAGENET1K_STD
from timm.data.transforms_factory import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms.v2 import Compose
from typing import Any

def get_default_image_transforms(
    resize_value: int | None = None,
    crop_value: int | None = None,
    random_erase: float | None = None,
    horizontal_flip: float | None = None,
    is_precropped: bool = False,
    autoaugment: bool = False,
) -> tuple[transforms_v2.Compose, transforms_v2.Compose]:
    """Constructs a default set of image transforms for training and validation.

    This is a generic function suitable for many image classification tasks.

    Args:
        resize_value (int | None): The size to resize the images to. Defaults to None.
        crop_value (int | None): The size to crop the images to. Defaults to None.
        random_erase (float | None): The probability of applying random erasing. Defaults to None.
        horizontal_flip (float | None): The probability of applying horizontal flip. Defaults to None.
        is_precropped (bool): Whether the images are cropped. Defaults to False.
        autoaugment (bool): Whether to use AutoAugment (True) or TrivialAugmentWide (False). Defaults to False.

    Returns:
        tuple[transforms_v2.Compose, transforms_v2.Compose]: Train and validation transforms.
    """
    train_transforms = []
    val_transforms = []

    if resize_value is not None:
        if is_precropped:
            train_transforms.append(
                transforms_v2.Resize(size=(resize_value, resize_value), antialias=True)
            )
            val_transforms.append(
                transforms_v2.Resize(size=(resize_value, resize_value), antialias=True)
            )
        else:
            train_transforms.append(
                transforms_v2.Resize(size=resize_value, antialias=True)
            )
            val_transforms.append(
                transforms_v2.Resize(size=resize_value, antialias=True)
            )

    if horizontal_flip is not None:
        train_transforms.append(transforms_v2.RandomHorizontalFlip(p=horizontal_flip))

    if crop_value is not None:
        train_transforms.append(transforms_v2.RandomCrop(size=crop_value))
        val_transforms.append(transforms_v2.CenterCrop(crop_value))
    
    if autoaugment:
        train_transforms.append(transforms_v2.AutoAugment(policy=transforms_v2.AutoAugmentPolicy.IMAGENET))
    else:
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


def get_deit_transforms(is_precropped: bool = False) -> tuple[Any, Any]:
    """
    DeiT-style transforms via timm.create_transform.
    If is_precropped is True (e.g., cropped CUB), keep eval crop_pct=1.0 to avoid 256->center-crop,
    and make train behave like a deterministic resize to 224 without random resized crop.
    """
    if is_precropped:
        # Training: disable random resized crop by forcing full-area, 1:1 ratio
        train_transform = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=0.3,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            scale=(1.0, 1.0),
            ratio=(1.0, 1.0),
        )
        # Validation: direct resize to 224 via crop_pct=1.0
        val_transform = create_transform(
            input_size=224,
            is_training=False,
            interpolation='bicubic',
            crop_pct=1.0,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
        return train_transform, val_transform

    # Default DeiT pipeline (non-cropped datasets)
    train_transform = create_transform(
        input_size=224,
        is_training=True,
        color_jitter=0.3,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )

    val_transform = create_transform(
        input_size=224,
        is_training=False,
        interpolation='bicubic',  # default crop_pct=0.875 -> 256->center-crop 224
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )

    return train_transform, val_transform
