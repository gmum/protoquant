import os
import pandas as pd
from torchvision.datasets import VisionDataset
import torchvision.io as io
from torch.utils.data import Dataset
from .imagenet import get_imagenet_transforms


class CUB200(VisionDataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.root = os.path.join(root, "CUB_200_2011")
        self.train = train

        # Load metadata
        image_paths = pd.read_csv(
            os.path.join(self.root, "images.txt"), sep=" ", names=["img_id", "path"]
        )
        labels = pd.read_csv(
            os.path.join(self.root, "image_class_labels.txt"),
            sep=" ",
            names=["img_id", "label"],
        )
        train_test_split = pd.read_csv(
            os.path.join(self.root, "train_test_split.txt"),
            sep=" ",
            names=["img_id", "is_train"],
        )

        # Merge metadata
        data = image_paths.merge(labels, on="img_id").merge(
            train_test_split, on="img_id"
        )

        # Select train or test split
        self.data = data[data["is_train"] == int(train)]

        # Convert labels to zero-based index
        self.data["label"] -= 1

        self.image_paths = self.data["path"].tolist()
        self.labels = self.data["label"].tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, "images", self.image_paths[index])
        label = self.labels[index]

        # Use torchvision.io for faster image loading
        image = io.read_image(img_path).float() / 255.0  # Normalize to [0,1]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def get_cub200(
    path: str,
    resize_value: int | None = None,
    crop_value: int | None = None,
    random_erase: float | None = None,
    horizontal_flip: float | None = None,
) -> tuple[Dataset, Dataset]:
    """Constructs the CUB200-2011 dataset.

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

    train_dataset = CUB200(
        root=path,
        train=True,
        transform=train_transform,
    )

    validate_dataset = CUB200(
        root=path,
        train=False,
        transform=test_transform,
    )

    return train_dataset, validate_dataset
