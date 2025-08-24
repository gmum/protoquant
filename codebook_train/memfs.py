from torchvision.datasets import ImageNet, CIFAR10, CIFAR100, Flowers102
from torchvision.datasets.imagenet import ARCHIVE_META
import sys
import argparse
import shutil
import pathlib
import logging

from PIL import Image
from os.path import join
import os
import scipy.io

import torch.utils.data as data
from torchvision.datasets.utils import download_url, list_dir

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


class StanfordCars:
    def __init__(self, target_dir: pathlib.Path, source_dir: pathlib.Path) -> None:
        self.target_dir = target_dir
        self.source_dir = source_dir

    def load_to_memfs(self):
        self.target_dir.mkdir(exist_ok=True)

        cub_source = self.source_dir / "stanford_cars"
        cub_target = self.target_dir / "stanford_cars"
        if not cub_source.exists():
            raise FileNotFoundError(
                f"Stanford Cars dataset not found at {cub_source}. Please ensure the dataset is downloaded and available."
            )

        logger.info(f"copying Stanford Cars dataset from {cub_source} to {cub_target}")

        # copy the directory
        shutil.copytree(
            cub_source,
            cub_target,
            dirs_exist_ok=True,
        )

        logger.info(f"done copying Stanford Cars dataset to: {self.target_dir}")


# source: https://github.com/zrsmithson/Stanford-dogs/blob/master/data/stanford_dogs_data.py
class StanfordDogs(data.Dataset):
    """`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        cropped (bool, optional): If true, the images will be cropped into the bounding box specified
            in the annotations
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset tar files from the internet and
            puts it in root directory. If the tar files are already downloaded, they are not
            downloaded again.
    """

    folder = "StanfordDogs"
    download_url_prefix = "http://vision.stanford.edu/aditya86/ImageNetDogs"

    def __init__(
        self,
        root,
        train=True,
        cropped=False,
        transform=None,
        target_transform=None,
        download=False,
    ):
        self.root = join(os.path.expanduser(root), self.folder)
        self.train = train
        self.cropped = cropped
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        split = self.load_split()

        self.images_folder = join(self.root, "Images")
        self.annotations_folder = join(self.root, "Annotation")
        self._breeds = list_dir(self.images_folder)

        if self.cropped:
            self._breed_annotations = [
                [
                    (annotation, box, idx)
                    for box in self.get_boxes(join(self.annotations_folder, annotation))
                ]
                for annotation, idx in split
            ]
            self._flat_breed_annotations = sum(self._breed_annotations, [])

            self._flat_breed_images = [
                (annotation + ".jpg", idx)
                for annotation, box, idx in self._flat_breed_annotations
            ]
        else:
            self._breed_images = [
                (annotation + ".jpg", idx) for annotation, idx in split
            ]

            self._flat_breed_images = self._breed_images

        self.classes = [
            "Chihuaha",
            "Japanese Spaniel",
            "Maltese Dog",
            "Pekinese",
            "Shih-Tzu",
            "Blenheim Spaniel",
            "Papillon",
            "Toy Terrier",
            "Rhodesian Ridgeback",
            "Afghan Hound",
            "Basset Hound",
            "Beagle",
            "Bloodhound",
            "Bluetick",
            "Black-and-tan Coonhound",
            "Walker Hound",
            "English Foxhound",
            "Redbone",
            "Borzoi",
            "Irish Wolfhound",
            "Italian Greyhound",
            "Whippet",
            "Ibizian Hound",
            "Norwegian Elkhound",
            "Otterhound",
            "Saluki",
            "Scottish Deerhound",
            "Weimaraner",
            "Staffordshire Bullterrier",
            "American Staffordshire Terrier",
            "Bedlington Terrier",
            "Border Terrier",
            "Kerry Blue Terrier",
            "Irish Terrier",
            "Norfolk Terrier",
            "Norwich Terrier",
            "Yorkshire Terrier",
            "Wirehaired Fox Terrier",
            "Lakeland Terrier",
            "Sealyham Terrier",
            "Airedale",
            "Cairn",
            "Australian Terrier",
            "Dandi Dinmont",
            "Boston Bull",
            "Miniature Schnauzer",
            "Giant Schnauzer",
            "Standard Schnauzer",
            "Scotch Terrier",
            "Tibetan Terrier",
            "Silky Terrier",
            "Soft-coated Wheaten Terrier",
            "West Highland White Terrier",
            "Lhasa",
            "Flat-coated Retriever",
            "Curly-coater Retriever",
            "Golden Retriever",
            "Labrador Retriever",
            "Chesapeake Bay Retriever",
            "German Short-haired Pointer",
            "Vizsla",
            "English Setter",
            "Irish Setter",
            "Gordon Setter",
            "Brittany",
            "Clumber",
            "English Springer Spaniel",
            "Welsh Springer Spaniel",
            "Cocker Spaniel",
            "Sussex Spaniel",
            "Irish Water Spaniel",
            "Kuvasz",
            "Schipperke",
            "Groenendael",
            "Malinois",
            "Briard",
            "Kelpie",
            "Komondor",
            "Old English Sheepdog",
            "Shetland Sheepdog",
            "Collie",
            "Border Collie",
            "Bouvier des Flandres",
            "Rottweiler",
            "German Shepard",
            "Doberman",
            "Miniature Pinscher",
            "Greater Swiss Mountain Dog",
            "Bernese Mountain Dog",
            "Appenzeller",
            "EntleBucher",
            "Boxer",
            "Bull Mastiff",
            "Tibetan Mastiff",
            "French Bulldog",
            "Great Dane",
            "Saint Bernard",
            "Eskimo Dog",
            "Malamute",
            "Siberian Husky",
            "Affenpinscher",
            "Basenji",
            "Pug",
            "Leonberg",
            "Newfoundland",
            "Great Pyrenees",
            "Samoyed",
            "Pomeranian",
            "Chow",
            "Keeshond",
            "Brabancon Griffon",
            "Pembroke",
            "Cardigan",
            "Toy Poodle",
            "Miniature Poodle",
            "Standard Poodle",
            "Mexican Hairless",
            "Dingo",
            "Dhole",
            "African Hunting Dog",
        ]

    def __len__(self):
        return len(self._flat_breed_images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, target_class = self._flat_breed_images[index]
        image_path = join(self.images_folder, image_name)
        image = Image.open(image_path).convert("RGB")

        if self.cropped:
            image = image.crop(self._flat_breed_annotations[index][1])

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target_class = self.target_transform(target_class)

        return image, target_class

    def download(self):
        import tarfile

        if os.path.exists(join(self.root, "Images")) and os.path.exists(
            join(self.root, "Annotation")
        ):
            if (
                len(os.listdir(join(self.root, "Images")))
                == len(os.listdir(join(self.root, "Annotation")))
                == 120
            ):
                print("Files already downloaded and verified")
                return

        for filename in ["images", "annotation", "lists"]:
            tar_filename = filename + ".tar"
            url = self.download_url_prefix + "/" + tar_filename
            download_url(url, self.root, tar_filename, None)
            print("Extracting downloaded file: " + join(self.root, tar_filename))
            with tarfile.open(join(self.root, tar_filename), "r") as tar_file:
                tar_file.extractall(self.root)
            os.remove(join(self.root, tar_filename))

    def load_split(self):
        if self.train:
            split = scipy.io.loadmat(join(self.root, "train_list.mat"))[
                "annotation_list"
            ]
            labels = scipy.io.loadmat(join(self.root, "train_list.mat"))["labels"]
        else:
            split = scipy.io.loadmat(join(self.root, "test_list.mat"))[
                "annotation_list"
            ]
            labels = scipy.io.loadmat(join(self.root, "test_list.mat"))["labels"]

        split = [item[0][0] for item in split]
        labels = [item[0] - 1 for item in labels]
        return list(zip(split, labels))

    def stats(self):
        counts = {}
        for index in range(len(self._flat_breed_images)):
            image_name, target_class = self._flat_breed_images[index]
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        print(
            "%d samples spanning %d classes (avg %f per class)"
            % (
                len(self._flat_breed_images),
                len(counts.keys()),
                float(len(self._flat_breed_images)) / float(len(counts.keys())),
            )
        )

        return counts


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
        elif args.dataset == "flowers102":
            logger.info("downloading flowers102")
            Flowers102(
                root=target_dir_path,
                split="train",
                download=True,
            )
            Flowers102(
                root=target_dir_path,
                split="val",
                download=True,
            )
        elif args.dataset == "stanford_cars":
            logger.info("loading stanford_cars")
            StanfordCars(
                target_dir=target_dir_path, source_dir=source_dir_path
            ).load_to_memfs()
        elif args.dataset == "stanford_dogs":
            logger.info("loading stanford_dogs")
            StanfordDogs(root=target_dir_path, download=True, train=True)
            StanfordDogs(root=target_dir_path, download=True, train=False)
        else:
            logger.error(
                f"Unsupported dataset: {args.dataset}. Supported datasets are: imagenet1k, cifar10, cifar100, cub200, stanford_cars, flowers102, stanford_dogs."
            )
            sys.exit(1)

        logger.info("Dataset loading completed successfully")
        sys.exit(0)  # Success
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
