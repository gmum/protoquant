import os
import pandas as pd
from torchvision.datasets import VisionDataset
from torchvision.transforms.v2 import Compose
from PIL import Image


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
        self.data = data[data["is_train"] == int(train)].copy()

        # Convert labels to zero-based index
        self.data.loc[:, "label"] -= 1

        self.image_paths = self.data["path"].tolist()
        self.labels = self.data["label"].tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, "images", self.image_paths[index])
        label = self.labels[index]

        # Use torchvision.io for faster image loading
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def get_cub200(
    path: str,
    train_transform: Compose,
    test_transform: Compose,
) -> tuple[CUB200, CUB200]:
    """Constructs the CUB200-2011 dataset.

    Args:
        path (str): Path to the dataset.
        train_transform (Compose): Transformations to apply to the training set.
        test_transform (Compose): Transformations to apply to the validation set.

    Returns:
        tuple[CUB200, CUB200]: Train and validation datasets.
    """

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
