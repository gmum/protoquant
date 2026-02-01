from torchvision.datasets import ImageNet
from torchvision.transforms.v2 import Compose


def get_imagenet1k(
    path: str,
    train_transform: Compose,
    test_transform: Compose,
) -> tuple[ImageNet, ImageNet]:
    """Constructs the ImageNet1K dataset.

    Args:
        path (str): Path to the dataset.
        train_transform (object): Transformations to apply to the training set.
        test_transform (object): Transformations to apply to the validation set.

    Returns:
        tuple[Dataset, Dataset]: Train and validation datasets.
    """

    train_dataset = ImageNet(
        root=path,
        split="train",
        transform=train_transform,
    )

    validate_dataset = ImageNet(
        root=path,
        split="val",
        transform=test_transform,
    )

    return train_dataset, validate_dataset
