# import dataloaders
from torchvision.datasets import StanfordCars
from torchvision.transforms.v2 import Compose


def get_stanford_cars(
    path: str,
    train_transform: Compose,
    test_transform: Compose,
) -> tuple[StanfordCars, StanfordCars]:
    """Constructs the Stanford Cars dataset.

    Args:
        path (str): Path to the dataset.
        train_transform (object): Transformations to apply to the training set.
        test_transform (object): Transformations to apply to the validation set.

    Returns:
        tuple[StanfordCars, StanfordCars]: Train and validation datasets.
    """

    train_dataset = StanfordCars(
        root=path, split="train", transform=train_transform, download=False
    )

    validate_dataset = StanfordCars(
        root=path,
        split="test",  # Stanford Cars uses 'test' for the validation split
        transform=test_transform,
        download=False,
    )

    return train_dataset, validate_dataset
