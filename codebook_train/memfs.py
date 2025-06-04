from torchvision.datasets import ImageNet, CIFAR10, CIFAR100
from torchvision.datasets.imagenet import ARCHIVE_META
import sys
import argparse
import shutil
import pathlib
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Imagenet:
    def __init__(self, target_dir: pathlib.Path, source_dir: pathlib.Path) -> None:
        self.target_dir = target_dir
        self.source_dir = source_dir

    def load_to_memfs(self):
        self.target_dir.mkdir(exist_ok=True)

        tar_path = self.source_dir / "ILSVRC2012_img_train.tar"
        if not tar_path.exists():
            raise FileNotFoundError(
                f"ImageNet training archive not found at {tar_path}. Please ensure the archive is downloaded and available (copying of the unpacked dataset is too slow)."
            )
            
        for tar_file, _ in ARCHIVE_META.values():
            (self.target_dir / tar_file).symlink_to(self.source_dir / tar_file)

        logger.info("unpacking train set")
        ImageNet(self.target_dir, split="train")
        logger.info("unpacking val set")
        ImageNet(self.target_dir, split="val")
        logger.info(f"done unpacking imagenet at: {self.target_dir}")


class CUB200:
    def __init__(self, target_dir: pathlib.Path, source_dir: pathlib.Path) -> None:
        self.target_dir = target_dir
        self.source_dir = source_dir

    def load_to_memfs(self):
        self.target_dir.mkdir(exist_ok=True)

        cub_source = self.source_dir / "CUB_200_2011"
        cub_target = self.target_dir / "CUB_200_2011"
        if not cub_source.exists():
            raise FileNotFoundError(
                f"CUB-200 dataset not found at {cub_source}. Please ensure the dataset is downloaded and available."
            )

        logger.info(f"copying CUB-200 dataset from {cub_source} to {cub_target}")

        # copy the directory
        shutil.copytree(
            cub_source,
            cub_target,
            dirs_exist_ok=True,
        )

        logger.info(f"done copying CUB-200 dataset to: {self.target_dir}")


def main(args):
    parser = argparse.ArgumentParser(description="Load dataset to memory file system.")
    parser.add_argument("--target_dir", type=str, required=True)
    parser.add_argument("--source_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    target_dir_path = pathlib.Path(args.target_dir)
    source_dir_path = pathlib.Path(args.source_dir)

    try:
        if args.dataset == "imagenet1k":
            Imagenet(target_dir_path, source_dir_path).load_to_memfs()
        elif args.dataset == "cifar10":
            logger.info("downloading cifar10")
            CIFAR10(
                root=target_dir_path,
                train=True,
                download=True,
            )
            CIFAR10(
                root=target_dir_path,
                train=False,
                download=True,
            )
        elif args.dataset == "cifar100":
            logger.info("downloading cifar100")
            CIFAR100(
                root=target_dir_path,
                train=True,
                download=True,
            )
            CIFAR100(
                root=target_dir_path,
                train=False,
                download=True,
            )
        elif args.dataset == "cub200":
            CUB200(target_dir_path, source_dir_path).load_to_memfs()
        else:
            logger.error(
                f"Unsupported dataset: {args.dataset}. Supported datasets are: imagenet1k, cifar10, cifar100, cub200."
            )
            sys.exit(1)

        logger.info("Dataset loading completed successfully")
        sys.exit(0)  # Success
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
