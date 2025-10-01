# import dataloaders
from torchvision.datasets import Flowers102
from src.datasets.transforms import get_default_image_transforms
from torchvision.transforms.v2 import Compose

def get_flowers102(
    path: str,
    train_transform: Compose,
    test_transform: Compose,
) -> tuple[Flowers102, Flowers102]:
    """Constructs the Flowers102 dataset.

    Args:
        path (str): Path to the dataset.
        train_transform (Compose): Transformations to apply to the training set.
        test_transform (Compose): Transformations to apply to the validation set.

    Returns:
        tuple[Flowers102, Flowers102]: Train and validation datasets.
    """

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
