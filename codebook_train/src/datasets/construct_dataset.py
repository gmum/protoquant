import torch

from src.config.main_config import MainConfig
from src.datasets.imagenet import get_imagenet1k
from src.datasets.cub import get_cub200
from torch.utils.data import DataLoader


def get_dataset(
    cfg: MainConfig,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    if cfg.dataset.name == "imagenet1k":
        return get_imagenet1k(
            cfg.dataset._path,
            resize_value=cfg.dataset.resize_size,
            crop_value=cfg.dataset.crop_size,
            random_erase=cfg.dataset.random_erase,
            horizontal_flip=cfg.dataset.horizontal_flip,
        )
    elif cfg.dataset.name == "cub200":
        return get_cub200(
            cfg.dataset._path,
            resize_value=cfg.dataset.resize_size,
            crop_value=cfg.dataset.crop_size,
            random_erase=cfg.dataset.random_erase,
            horizontal_flip=cfg.dataset.horizontal_flip,
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset.name}")


def get_dataloaders(
    cfg: MainConfig,
) -> tuple[DataLoader, DataLoader]:
    train_dataset, validate_dataset = get_dataset(cfg)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_dataloader.batch_size,
        shuffle=True,
        pin_memory=cfg.train_dataloader.pin_memory,
        num_workers=cfg.train_dataloader.num_workers,
        drop_last=cfg.train_dataloader.drop_last,
    )
    validation_loader = DataLoader(
        validate_dataset,
        batch_size=cfg.val_dataloader.batch_size,
        shuffle=False,
        pin_memory=cfg.val_dataloader.pin_memory,
        num_workers=cfg.val_dataloader.num_workers,
        drop_last=cfg.val_dataloader.drop_last,
    )

    return train_loader, validation_loader
