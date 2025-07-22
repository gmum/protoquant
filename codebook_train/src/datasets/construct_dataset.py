from src.config.main_config import MainConfig
from src.datasets.imagenet import get_imagenet1k
from src.datasets.cub import get_cub200
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from src.datasets.stanford_cars import get_stanford_cars
from src.datasets.flowers102 import get_flowers102

from torchvision.datasets import StanfordCars, Flowers102, ImageNet
from src.datasets.cub import CUB200
from src.datasets.stanford_dogs import StanfordDogs, get_stanford_dogs

AVAILABLE_DATASET = CUB200 | StanfordCars | Flowers102 | ImageNet | StanfordDogs

def get_dataset(
    cfg: MainConfig,
) -> tuple[AVAILABLE_DATASET, AVAILABLE_DATASET]:
    print("XDDDDDD: ", type(cfg.dataset))
    if cfg.dataset.name == "imagenet1k":
        return get_imagenet1k(
            path=cfg.dataset._path,
            crop_value=cfg.dataset.crop_size,
            resize_value=cfg.dataset.resize_size,
            random_erase=cfg.dataset.random_erase,
            horizontal_flip=cfg.dataset.horizontal_flip,
        )
    elif cfg.dataset.name == "cub200":
        return get_cub200(
            path=cfg.dataset._path,
            crop_value=cfg.dataset.crop_size,
            resize_value=cfg.dataset.resize_size,
            random_erase=cfg.dataset.random_erase,
            horizontal_flip=cfg.dataset.horizontal_flip,
        )
    elif cfg.dataset.name == "stanford_cars":
        return get_stanford_cars(
            path=cfg.dataset._path,
            crop_value=cfg.dataset.crop_size,
            resize_value=cfg.dataset.resize_size,
            random_erase=cfg.dataset.random_erase,
            horizontal_flip=cfg.dataset.horizontal_flip,
        )
    elif cfg.dataset.name == "flowers102":
        return get_flowers102(
            path=cfg.dataset._path,
            crop_value=cfg.dataset.crop_size,
            resize_value=cfg.dataset.resize_size,
            random_erase=cfg.dataset.random_erase,
            horizontal_flip=cfg.dataset.horizontal_flip,
        )
    elif cfg.dataset.name == "stanford_dogs":
        return get_stanford_dogs(
            path=cfg.dataset._path,
            crop_value=cfg.dataset.crop_size,
            resize_value=cfg.dataset.resize_size,
            random_erase=cfg.dataset.random_erase,
            horizontal_flip=cfg.dataset.horizontal_flip,
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")


def get_dataloaders(
    cfg: MainConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    train_sampler: DistributedSampler | None = None,
    val_sampler: DistributedSampler | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Constructs the dataloaders for training and validation datasets.

    Args:
        cfg (MainConfig):  Main configuration object
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset):  The validation dataset.
        train_sampler (DistributedSampler | None, optional): Training sampler for distributed training. Defaults to None.
        val_sampler (DistributedSampler | None, optional): Validation sampler for distributed training. Defaults to None.

    Returns:
        tuple[DataLoader, DataLoader]: _description_
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_dataloader.batch_size,
        shuffle=None if train_sampler else True,
        pin_memory=cfg.train_dataloader.pin_memory,
        num_workers=cfg.train_dataloader.num_workers,
        drop_last=cfg.train_dataloader.drop_last,
        persistent_workers=True,
        sampler=train_sampler,
    )

    validation_loader = DataLoader(
        val_dataset,
        batch_size=cfg.val_dataloader.batch_size,
        shuffle=None if val_sampler else False,
        pin_memory=cfg.val_dataloader.pin_memory,
        num_workers=cfg.val_dataloader.num_workers,
        drop_last=cfg.val_dataloader.drop_last,
        persistent_workers=True,
        sampler=val_sampler,
    )

    return train_loader, validation_loader


