import os
import argparse
import shutil
import pandas as pd
from PIL import Image
from tqdm import tqdm


def create_cropped_dataset(input_dir: str, output_dir: str):
    """
    Creates a new version of the CUB-200-2011 dataset where each image
    is cropped to its provided bounding box.

    Args:
        input_dir (str): The root directory of the original CUB_200_2011 dataset.
        output_dir (str): The directory where the new cropped dataset will be saved.
    """
    # Validate input directory
    if not os.path.basename(input_dir) == "CUB_200_2011":
        print(f"Warning: Input directory '{input_dir}' is not named 'CUB_200_2011'.")
        print("Please ensure it's the correct root directory.")

    images_txt_path = os.path.join(input_dir, "images.txt")
    bboxes_txt_path = os.path.join(input_dir, "bounding_boxes.txt")

    if not all(os.path.exists(p) for p in [images_txt_path, bboxes_txt_path]):
        raise FileNotFoundError(
            "Could not find 'images.txt' or 'bounding_boxes.txt' in the input directory. "
            "Please provide the correct path to the CUB_200_2011 dataset."
        )

    print(f"Reading data from: {input_dir}")
    print(f"Saving cropped dataset to: {output_dir}")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Load and merge metadata ---
    image_paths_df = pd.read_csv(images_txt_path, sep=" ", names=["img_id", "path"])
    bboxes_df = pd.read_csv(
        bboxes_txt_path, sep=" ", names=["img_id", "x", "y", "width", "height"]
    )

    # Merge to get a single DataFrame with path and bbox for each image
    data_df = image_paths_df.merge(bboxes_df, on="img_id")
    print(f"Found {len(data_df)} images to process.")

    # --- 2. Iterate, crop, and save images ---
    for index, row in tqdm(
        data_df.iterrows(), total=len(data_df), desc="Cropping images"
    ):
        # Construct source and destination paths
        src_path = os.path.join(input_dir, "images", row["path"])
        dest_path = os.path.join(output_dir, "images", row["path"])

        # Create destination subdirectory if it doesn't exist
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # Load the image
        with Image.open(src_path) as img:
            # Ensure image is in RGB format, as some might be grayscale
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Get bounding box coordinates
            x, y = int(row["x"]), int(row["y"])
            width, height = int(row["width"]), int(row["height"])

            # PIL's crop uses (left, upper, right, lower)
            # CUB provides (x, y, width, height) where x,y is the top-left corner
            left = x
            upper = y
            right = x + width
            lower = y + height

            # Crop the image
            img_cropped = img.crop((left, upper, right, lower))

            # Save the cropped image
            img_cropped.save(dest_path)

    # --- 3. Copy all metadata .txt files ---
    print("Copying metadata files...")
    files_to_copy = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    for file_name in files_to_copy:
        src_file = os.path.join(input_dir, file_name)
        dest_file = os.path.join(output_dir, file_name)
        shutil.copy(src_file, dest_file)

    print("\nDataset cropping complete!")
    print(f"New cropped dataset is ready at: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a cropped version of the CUB-200-2011 dataset.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Example usage:\n"
            "python crop_cub200.py "
            "--input_dir /path/to/your/datasets/CUB_200_2011 "
            "--output_dir /path/to/your/datasets/CUB_200_2011_Cropped"
        ),
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the root directory of the original CUB_200_2011 dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path where the new cropped dataset will be saved.",
    )

    args = parser.parse_args()
    create_cropped_dataset(args.input_dir, args.output_dir)
