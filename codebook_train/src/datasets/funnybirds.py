import json
import os
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.v2 import Compose


class FunnyBirdsClassification(Dataset[tuple[torch.Tensor, int]]):
    """FunnyBirds dataset wrapper compatible with the `src` training loops.

    The official framework returns a dict with keys like `image` and `class_idx`.
    Our training loops expect `(image, label)`.

    Images are stored as PNG with alpha; we drop the alpha channel exactly like
    the official FunnyBirds framework (ToTensor()[:-1]).
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        transform: Any | None = None,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError("FunnyBirds split must be 'train' or 'test'")

        resolved_root_dir = root_dir
        self.split = split
        self.transform = transform

        json_path = os.path.join(resolved_root_dir, f"dataset_{self.split}.json")
        if not os.path.isfile(json_path):
            candidate = os.path.join(resolved_root_dir, "FunnyBirds")
            candidate_json = os.path.join(candidate, f"dataset_{self.split}.json")
            if os.path.isfile(candidate_json):
                resolved_root_dir = candidate
                json_path = candidate_json

        self.root_dir = resolved_root_dir

        try:
            with open(json_path, "r") as f:
                self.params: list[dict[str, Any]] = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "FunnyBirds metadata JSON not found. Expected either "
                f"{os.path.join(root_dir, f'dataset_{split}.json')} or "
                f"{os.path.join(root_dir, 'FunnyBirds', f'dataset_{split}.json')}."
            ) from e

        # Keep the same ToTensor() behavior as the official code
        self._to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.params)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        class_idx = int(self.params[idx]["class_idx"])

        img_path = os.path.join(
            self.root_dir,
            self.split,
            str(class_idx),
            f"{idx:06d}.png",
        )

        image_pil = Image.open(img_path)
        image = self._to_tensor(image_pil)[:-1, :, :]  # drop alpha

        if self.transform is not None:
            image = self.transform(image)

        return image, class_idx


def get_funnybirds(
    path: str,
    train_transform: Compose,
    val_transform: Compose,
) -> tuple[FunnyBirdsClassification, FunnyBirdsClassification]:
    """Construct FunnyBirds train/test datasets.

    The official framework uses `train` and `test` splits.
    """

    train_ds = FunnyBirdsClassification(root_dir=path, split="train", transform=train_transform)
    val_ds = FunnyBirdsClassification(root_dir=path, split="test", transform=val_transform)
    return train_ds, val_ds
