#!/usr/bin/env python
"""
Converts CUB-200-2011 dataset from its original format to ImageFolder format
with train/ and test/ subdirectories.

Usage:
    python split_cub_to_imagefolder.py \
        --input_dir /path/to/cub200_cropped \
        --output_dir /path/to/cub200_imagefolder

This creates:
    output_dir/
    ├── train/
    │   ├── 001.Black_footed_Albatross/
    │   │   ├── Black_Footed_Albatross_0001_796111.jpg
    │   │   └── ...
    │   └── ...
    └── test/
        ├── 001.Black_footed_Albatross/
        └── ...
"""

import os
import argparse
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def split_cub_to_imagefolder(input_dir: str, output_dir: str, copy: bool = True):
    """
    Splits CUB-200-2011 into train/test ImageFolder structure.
    
    Args:
        input_dir: Path to CUB dataset (should contain CUB_200_2011/ subfolder or be the CUB_200_2011 folder itself)
        output_dir: Path where train/ and test/ folders will be created
        copy: If True, copy files. If False, create symlinks (faster, saves space)
    """
    # Handle both cases: input_dir is parent or CUB_200_2011 itself
    if os.path.basename(input_dir) == "CUB_200_2011":
        cub_root = input_dir
    elif os.path.exists(os.path.join(input_dir, "CUB_200_2011")):
        cub_root = os.path.join(input_dir, "CUB_200_2011")
    elif os.path.exists(os.path.join(input_dir, "images.txt")):
        # It's the CUB folder but not named CUB_200_2011
        cub_root = input_dir
    else:
        raise FileNotFoundError(
            f"Could not find CUB-200-2011 dataset in {input_dir}. "
            "Expected either 'CUB_200_2011/' subfolder or images.txt file."
        )
    
    # Check required files exist
    images_txt = os.path.join(cub_root, "images.txt")
    split_txt = os.path.join(cub_root, "train_test_split.txt")
    images_dir = os.path.join(cub_root, "images")
    
    if not all(os.path.exists(p) for p in [images_txt, split_txt, images_dir]):
        raise FileNotFoundError(
            f"Missing required files in {cub_root}. "
            "Need: images.txt, train_test_split.txt, and images/ folder."
        )
    
    print(f"Reading CUB-200-2011 from: {cub_root}")
    print(f"Output directory: {output_dir}")
    
    # Load metadata
    image_paths_df = pd.read_csv(images_txt, sep=" ", names=["img_id", "path"])
    split_df = pd.read_csv(split_txt, sep=" ", names=["img_id", "is_train"])
    
    # Merge
    data = image_paths_df.merge(split_df, on="img_id")
    
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Process each image
    train_count, test_count = 0, 0
    
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Splitting dataset"):
        img_path = row["path"]  # e.g., "001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg"
        is_train = row["is_train"] == 1
        
        # Extract class folder and filename
        class_folder = os.path.dirname(img_path)
        filename = os.path.basename(img_path)
        
        # Source path
        src = os.path.join(images_dir, img_path)
        
        # Destination path
        if is_train:
            dest_dir = os.path.join(train_dir, class_folder)
            train_count += 1
        else:
            dest_dir = os.path.join(test_dir, class_folder)
            test_count += 1
        
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, filename)
        
        # Copy or symlink
        if not os.path.exists(dest):
            if copy:
                shutil.copy2(src, dest)
            else:
                os.symlink(os.path.abspath(src), dest)
    
    print(f"\nDone!")
    print(f"  Train images: {train_count}")
    print(f"  Test images:  {test_count}")
    print(f"\nImageFolder structure created at: {output_dir}")
    print(f"  - {train_dir}")
    print(f"  - {test_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert CUB-200-2011 to ImageFolder format with train/test split.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to CUB dataset (containing CUB_200_2011/ or the folder itself)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path where train/ and test/ folders will be created",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symlinks instead of copying files (faster, saves disk space)",
    )
    
    args = parser.parse_args()
    split_cub_to_imagefolder(args.input_dir, args.output_dir, copy=not args.symlink)
