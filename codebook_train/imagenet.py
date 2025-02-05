from torchvision import transforms

# import dataloaders
from torchvision import datasets
from torch.utils.data import DataLoader

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

VAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def get_imagenet(
    data_path: str, batch_size: int, num_workers: int
) -> tuple[DataLoader, DataLoader]:
    imagenet_val_ds = datasets.ImageNet(
        root=data_path, split="val", transform=VAL_TRANSFORM
    )

    imagenet_train_ds = datasets.ImageNet(
        root=data_path, split="train", transform=TRAIN_TRANSFORM
    )

    val_loader = DataLoader(
        dataset=imagenet_val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    train_loader = DataLoader(
        dataset=imagenet_train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    return train_loader, val_loader
