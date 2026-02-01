import torch
from torchvision.transforms import v2 as transforms_v2
from src.config.constants import IMAGENET1K_MEAN, IMAGENET1K_STD
from timm.data.transforms_factory import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms.v2 import Compose
from typing import Any


def get_raw_tensor_transforms(resize: int | None = None) -> tuple[Compose, Compose]:
    """Return minimal transforms for datasets that already provide tensor inputs.

    This is useful for benchmarks like FunnyBirds where the official protocol uses
    no normalization and no augmentation.

    Args:
        resize: If provided, resize images to this size (e.g., 224 for ViT models).
    """
    transform_list = []
    if resize is not None:
        transform_list.append(transforms_v2.Resize((resize, resize)))
    transform_list.append(transforms_v2.ToDtype(torch.float32, scale=False))

    transform = transforms_v2.Compose(transform_list)
    return transform, transform


def get_resize_normalize_transforms(size: int = 224) -> tuple[Compose, Compose]:
    """Resize to a fixed size and apply ImageNet normalization.

    Args:
        size: Output size for both height and width.

    Returns:
        tuple[Compose, Compose]: Train and validation transforms (identical).
    """
    transform = transforms_v2.Compose(
        [
            transforms_v2.Resize(size=(size, size), antialias=True),
            transforms_v2.ToImage(),
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Normalize(mean=IMAGENET1K_MEAN, std=IMAGENET1K_STD),
        ]
    )
    return transform, transform


def get_transforms_by_mode(
    mode: str,
    *,
    model_name: str | None = None,
    resize_size: int | None = 256,
    crop_size: int | None = 224,
    random_erase: float | None = 0.1,
    horizontal_flip: float | None = 0.5,
    is_precropped: bool = False,
    autoaugment: bool = False,
) -> tuple[Compose, Compose]:
    """Select a transform pipeline by mode.

    Args:
        mode: One of "default", "deit", "raw", or "resize_norm".
        model_name: Optional model name for raw transform resizing (ViT -> 224).
        resize_size: Resize value for default transforms.
        crop_size: Crop value for default transforms.
        random_erase: Random erasing probability for default transforms.
        horizontal_flip: Random horizontal flip probability for default transforms.
        is_precropped: Whether images are precropped.
        autoaugment: Whether to use AutoAugment for default transforms.

    Returns:
        tuple[Compose, Compose]: Train and validation transforms.
    """
    normalized_mode = mode.lower()
    if normalized_mode == "raw":
        resize = 224 if model_name and "vit" in model_name.lower() else None
        return get_raw_tensor_transforms(resize=resize)
    if normalized_mode == "deit":
        return get_deit_transforms(is_precropped=is_precropped)
    if normalized_mode in {"resize_norm", "resize_normalize", "resize"}:
        return get_resize_normalize_transforms(size=224)
    if normalized_mode == "default":
        return get_default_image_transforms(
            resize_value=resize_size,
            crop_value=crop_size,
            random_erase=random_erase,
            horizontal_flip=horizontal_flip,
            is_precropped=is_precropped,
            autoaugment=autoaugment,
        )

    raise ValueError(
        f"Unknown transforms mode '{mode}'. Expected one of: default, deit, raw, resize_norm."
    )


def get_transforms_from_config(
    dataset_cfg: Any,
    *,
    model_name: str | None = None,
    mode: str | None = None,
) -> tuple[Compose, Compose]:
    """Select transforms using dataset config flags or an explicit mode."""
    selected_mode = mode
    if selected_mode is None:
        if dataset_cfg.use_raw_transforms:
            selected_mode = "raw"
        elif dataset_cfg.use_deit_transforms:
            selected_mode = "deit"
        elif dataset_cfg.use_resize_norm_transforms:
            selected_mode = "resize_norm"
        else:
            selected_mode = "default"

    return get_transforms_by_mode(
        selected_mode,
        model_name=model_name,
        resize_size=dataset_cfg.resize_size,
        crop_size=dataset_cfg.crop_size,
        random_erase=dataset_cfg.random_erase,
        horizontal_flip=dataset_cfg.horizontal_flip,
        is_precropped=dataset_cfg.is_precropped,
        autoaugment=dataset_cfg.autoaugment,
    )


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
        train_transforms.append(
            transforms_v2.AutoAugment(policy=transforms_v2.AutoAugmentPolicy.IMAGENET)
        )
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
            auto_augment="rand-m9-mstd0.5-inc1",
            interpolation="bicubic",
            re_prob=0.25,
            re_mode="pixel",
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
            interpolation="bicubic",
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
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation="bicubic",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )

    val_transform = create_transform(
        input_size=224,
        is_training=False,
        interpolation="bicubic",  # default crop_pct=0.875 -> 256->center-crop 224
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )

    return train_transform, val_transform
