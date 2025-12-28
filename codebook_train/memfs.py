#!/usr/bin/env python

import sys
import argparse
import shutil
import pathlib
import logging
import tarfile
import os

from torchvision.datasets import CIFAR10, CIFAR100, Flowers102, StanfordCars
from torchvision.datasets.imagenet import ImageNet, ARCHIVE_META
from torchvision.datasets.utils import download_and_extract_archive, download_url

# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Helper ---
def copy_dataset(source_path: pathlib.Path, target_path: pathlib.Path):
    """Copies a dataset directory from source to target."""
    logger.info(f"Dataset found at source: {source_path}")
    logger.info(f"Copying to target: {target_path}...")
    shutil.copytree(source_path, target_path, dirs_exist_ok=True)
    logger.info("Copying complete.")


# --- Dataset Preparation Logic ---

def prepare_imagenet(target_dir: pathlib.Path, source_dir: pathlib.Path):
    """
    Prepares ImageNet by symlinking TARs from source and unpacking them in target.
    Warning: This process is very slow and requires significant disk space.
    """
    if (target_dir / "train").is_dir() and (target_dir / "val").is_dir():
        logger.info(f"ImageNet already prepared in {target_dir}.")
        return

    logger.warning("Preparing ImageNet. This will be very slow and consume >300GB disk space.")
    target_dir.mkdir(exist_ok=True, parents=True)

    train_tar_path = source_dir / "ILSVRC2012_img_train.tar"
    if not train_tar_path.exists():
        raise FileNotFoundError(f"ImageNet training archive not found at {train_tar_path}.")

    for tar_file, _ in ARCHIVE_META.values():
        source_file, target_file = source_dir / tar_file, target_dir / tar_file
        if not source_file.exists():
             raise FileNotFoundError(f"Required ImageNet archive not found: {source_file}")
        if not target_file.exists():
            target_file.symlink_to(source_file)

    logger.info("Symlinks created. Unpacking train set (this may take hours)...")
    ImageNet(str(target_dir), split="train")
    logger.info("Unpacking val set...")
    ImageNet(str(target_dir), split="val")
    logger.info(f"Done unpacking ImageNet at: {target_dir}")


def prepare_cifar(target_dir: pathlib.Path, source_dir: pathlib.Path, version: int):
    """Prepares CIFAR10 or CIFAR100."""
    dataset_class = CIFAR10 if version == 10 else CIFAR100
    folder_name = dataset_class.base_folder
    target_path, source_path = target_dir / folder_name, source_dir / folder_name

    if target_path.is_dir():
        logger.info(f"CIFAR-{version} already exists in {target_dir}.")
        return

    logger.info(f"Dataset not found in target, checking source dir: {source_dir}")
    if source_path.is_dir():
        copy_dataset(source_path, target_path)
    else:
        logger.info(f"CIFAR-{version} not found in source. Downloading...")
        dataset_class(root=str(target_dir), train=True, download=True)
        dataset_class(root=str(target_dir), train=False, download=True)
    logger.info(f"CIFAR-{version} is ready in {target_dir}.")


def prepare_flowers102(target_dir: pathlib.Path, source_dir: pathlib.Path):
    """Prepares the Flowers102 dataset."""
    # Check for the final unpacked image folder
    if (target_dir / "flowers-102" / "jpg").is_dir():
        logger.info(f"Flowers102 already exists in {target_dir}.")
        return

    logger.info(f"Dataset not found in target, checking source dir: {source_dir}")
    if (source_dir / "flowers-102").is_dir():
        copy_dataset(source_dir / "flowers-102", target_dir / "flowers-102")
    else:
        logger.info("Flowers102 not found in source. Downloading...")
        Flowers102(root=str(target_dir), split="train", download=True)
        Flowers102(root=str(target_dir), split="val", download=True)
        Flowers102(root=str(target_dir), split="test", download=True)
    logger.info(f"Flowers102 is ready in {target_dir}.")


def prepare_cub200(target_dir: pathlib.Path, source_dir: pathlib.Path):
    """Prepares the CUB-200-2011 dataset."""
    folder_name = "CUB_200_2011"
    target_path, source_path = target_dir / folder_name, source_dir / folder_name

    if target_path.is_dir():
        logger.info(f"CUB-200-2011 already exists in {target_dir}.")
        return

    logger.info(f"Dataset not found in target, checking source dir: {source_dir}")
    if source_path.is_dir():
        copy_dataset(source_path, target_path)
    else:
        logger.info("CUB-200-2011 not found in source. Downloading...")
        url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
        download_and_extract_archive(url, str(target_dir), filename="CUB_200_2011.tgz")
    logger.info(f"CUB-200-2011 is ready in {target_dir}.")


def prepare_stanford_dogs(target_dir: pathlib.Path, source_dir: pathlib.Path):
    """Prepares the Stanford Dogs dataset."""
    folder_name = "StanfordDogs"
    target_path, source_path = target_dir / folder_name, source_dir / folder_name

    if target_path.is_dir():
        logger.info(f"Stanford Dogs already exists in {target_dir}.")
        return

    logger.info(f"Dataset not found in target, checking source dir: {source_dir}")
    if source_path.is_dir():
        copy_dataset(source_path, target_path)
    else:
        logger.info("Stanford Dogs not found in source. Downloading...")
        for part in ["images", "annotation", "lists"]:
            filename = f"{part}.tar"
            url = f"http://vision.stanford.edu/aditya86/ImageNetDogs/{filename}"
            download_url(url, str(target_dir), filename)
            with tarfile.open(target_dir / filename, "r") as tar_ref:
                tar_ref.extractall(target_dir)
            os.remove(target_dir / filename)
        # Create the unified StanfordDogs directory
        os.rename(target_dir / "Images", target_path)
        for item in ["Annotation", "lists"]:
            shutil.move(str(target_dir / item), str(target_path / item))

    logger.info(f"Stanford Dogs is ready in {target_dir}.")


def prepare_stanford_cars(target_dir: pathlib.Path, source_dir: pathlib.Path):
    """Prepares the Stanford Cars dataset."""
    folder_name = "stanford_cars"
    target_path, source_path = target_dir / folder_name, source_dir / folder_name

    if target_path.is_dir():
        logger.info(f"Stanford Cars already exists in {target_dir}.")
        return

    logger.info(f"Dataset not found in target, checking source dir: {source_dir}")
    if source_path.is_dir():
        copy_dataset(source_path, target_path)
    else:
        logger.info("Stanford Cars not found in source. Downloading...")
        # Download and extract train/test sets
        StanfordCars(root=str(target_dir), split="train", download=True)
        StanfordCars(root=str(target_dir), split="test", download=True)

        # Merge them into the expected single 'stanford_cars' directory
        logger.info("Merging downloaded train and test sets...")
        (target_dir / "cars_train").rename(target_path)
        shutil.copytree(target_dir / "cars_test", target_path, dirs_exist_ok=True)
        shutil.rmtree(target_dir / "cars_test")
        shutil.rmtree(target_dir / "stanford_cars_annos")
        # remove tgz files
        os.remove(target_dir / "cars_test.tgz")
        os.remove(target_dir / "cars_train.tgz")
        os.remove(target_dir / "car_devkit.tgz")

    logger.info(f"Stanford Cars is ready in {target_dir}.")


def prepare_funnybirds(target_dir: pathlib.Path, source_dir: pathlib.Path):
    """Prepares the FunnyBirds dataset.

    The FunnyBirds framework expects a directory structure like:
      FunnyBirds/
        dataset_train.json
        dataset_test.json
        classes.json
        parts.json
        train/<class_idx>/<000000.png>
        test/<class_idx>/<000000.png>

    This handler copies an existing FunnyBirds dataset from `source_dir`.
    Downloads are intentionally disabled (cluster-friendly / reproducible).
    """
    folder_name = "FunnyBirds"
    target_path, source_path = target_dir / folder_name, source_dir / folder_name

    required_files = [
        "dataset_train.json",
        "dataset_test.json",
        "classes.json",
        "parts.json",
    ]

    def _has_required_files(path: pathlib.Path) -> bool:
        return all((path / fname).is_file() for fname in required_files)

    # A simple, robust "already prepared" check
    if _has_required_files(target_path):
        logger.info(f"FunnyBirds already exists in {target_path}.")
        return

    logger.info(f"Dataset not found in target, checking source dir: {source_dir}")

    # Support both common source layouts:
    # 1) source_dir/FunnyBirds/<files>
    # 2) source_dir/<files>
    if _has_required_files(source_path):
        copy_dataset(source_path, target_path)
        logger.info(f"FunnyBirds is ready in {target_path}.")
        return

    if _has_required_files(source_dir):
        copy_dataset(source_dir, target_path)
        logger.info(f"FunnyBirds is ready in {target_path}.")
        return

    raise FileNotFoundError(
        "FunnyBirds not found in source_dir and downloads are disabled. "
        f"Expected either: {source_path} or {source_dir} (containing {', '.join(required_files)})."
    )


# --- Main Dispatcher ---

def main():
    parser = argparse.ArgumentParser(
        description="Prepare a dataset by copying from a source directory or downloading.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--target_dir", type=str, required=True, help="Directory to place the final dataset.")
    parser.add_argument("--source_dir", type=str, required=True, help="Directory with existing datasets to copy.")
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=[
            "imagenet1k", "cifar10", "cifar100", "cub200",
            "stanford_cars", "flowers102", "stanford_dogs", "funnybirds"
        ],
        help="Name of the dataset to prepare.",
    )
    args = parser.parse_args()

    target_dir = pathlib.Path(args.target_dir).resolve()
    source_dir = pathlib.Path(args.source_dir).resolve()
    target_dir.mkdir(exist_ok=True, parents=True)
    source_dir.mkdir(exist_ok=True, parents=True)

    dataset_handlers = {
        "imagenet1k": prepare_imagenet,
        "cifar10": lambda t, s: prepare_cifar(t, s, version=10),
        "cifar100": lambda t, s: prepare_cifar(t, s, version=100),
        "cub200": prepare_cub200,
        "stanford_cars": prepare_stanford_cars,
        "flowers102": prepare_flowers102,
        "stanford_dogs": prepare_stanford_dogs,
        "funnybirds": prepare_funnybirds,
    }

    try:
        handler = dataset_handlers[args.dataset]
        handler(target_dir, source_dir)
        logger.info(f"Successfully prepared dataset '{args.dataset}' in {target_dir}")
        sys.exit(0)
    except Exception:
        logger.exception(f"An error occurred while preparing dataset '{args.dataset}'")
        sys.exit(1)


if __name__ == "__main__":
    main()