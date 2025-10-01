import torch
from torchvision.transforms import v2 as transforms_v2
from src.config.constants import IMAGENET1K_MEAN, IMAGENET1K_STD
from timm.data.transforms_factory import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms.v2 import Compose

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


def get_deit_transforms() -> tuple[Compose, Compose]:
    """
    Creates and returns the specific data augmentation pipelines
    function used for training the DeiT (Data-efficient Image Transformer) model.

    This function hardcodes the parameters from the official DeiT repository to
    ensure faithful replication. It uses the `timm` library, as the original
    implementation does.

    Returns:
        tuple[Compose, Compose]: A tuple containing:
            - train_transform (Compose): The transformation pipeline for the training set.
            - val_transform (Compose): The transformation pipeline for the validation set.
    """
    # --- 1. Training Transformations ---
    # These are the default augmentation parameters from the DeiT repository's main.py
    train_transform = create_transform(
        input_size=224,
        is_training=True,
        color_jitter=0.3,
        auto_augment='rand-m9-mstd0.5-inc1',  # RandAugment
        interpolation='bicubic',
        re_prob=0.25,  # Random Erasing probability
        re_mode='pixel',
        re_count=1,
    )

    # --- 2. Validation Transformations ---
    # This logic mirrors the `build_transform` function in the DeiT repo for validation
    val_transform = transforms_v2.Compose([
        # The resize size is calculated as int(224 / 0.875), which equals 256
        transforms_v2.Resize(256, interpolation=transforms_v2.InterpolationMode.BICUBIC),
        transforms_v2.CenterCrop(224),
        transforms_v2.ToTensor(),
        transforms_v2.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    return train_transform, val_transform
