# import dataloaders
from torchvision.datasets import Flowers102
from src.datasets.transforms import get_default_image_transforms


def get_flowers102(
    path: str,
    resize_value: int | None = None,
    crop_value: int | None = None,
    random_erase: float | None = None,
    horizontal_flip: float | None = None,
    is_precropped: bool = False,
) -> tuple[Flowers102, Flowers102]:
    """Constructs the Flowers102 dataset.

    Args:
        path (str): Path to the dataset.
        resize_value (int | None): The size to resize the images to. Defaults to None.
        crop_value (int | None): The size to crop the images to. Defaults to None.
        random_erase (float | None): The probability of applying random erasing. Defaults to None.
        horizontal_flip (float | None): The probability of applying horizontal flip. Defaults to None.
        is_precropped (bool): Whether the images are pre-cropped. Defaults to False.

    Returns:
        tuple[Flowers102, Flowers102]: Train and validation datasets.
    """
    train_transform, test_transform = get_default_image_transforms(
        resize_value=resize_value,
        crop_value=crop_value,
        random_erase=random_erase,
        horizontal_flip=horizontal_flip,
        is_precropped=is_precropped,
    )

    train_dataset = Flowers102(
        root=path, split="train", transform=train_transform, download=True
    )

    validate_dataset = Flowers102(
        root=path,
        split="val",  # Flowers102 has a 'val' split
        transform=test_transform,
        download=True,
    )

    return train_dataset, validate_dataset
