import torch
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from argparse import Namespace

import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from torchvision.transforms import v2 as transforms_v2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# --- Prerequisite Imports ---
# These are still needed as they contain the core logic for your model/data handling.
from src.datasets.cub import CUB200
from purity_benchmark.config import PurityBenchConfig
from src.pipnet_utils import build_pipnet_model
from src.datasets.construct_dataset import get_dataset
from purity_benchmark.func import get_patch_size
from purity_benchmark.vis_pipnet import get_img_coordinates


def generate_purity_csv(
    model: torch.nn.Module,
    projectloader: torch.utils.data.DataLoader[CUB200],
    cfg: PurityBenchConfig,
    device: torch.device,
) -> Path:
    """Generates a CSV file with coordinates of top-k patches for each prototype."""
    model.eval()
    
    dataset = projectloader.dataset
    image_root = Path(dataset.root) / "images"
    # The `(str(...),)` creates a tuple to match the (path, label) structure
    # that the rest of the function expects, even though we don't use the label.
    imgs = [(str(image_root / rel_path),) for rel_path in dataset.image_paths]

    classification_weights = model.head.classifier.weight
    num_prototypes = model.head.P
    
    try:
        # Get a single batch from the dataloader to perform a forward pass
        sample_input, _ = next(iter(projectloader))
        sample_input = sample_input.to(device)
        with torch.no_grad():
            output = model(sample_input)
            # The shape is (batch, protos, width, height)
            wshape = output.proto_fmap.shape[-1]
        print(f"Dynamically determined wshape: {wshape}")
    except StopIteration:
        print("DataLoader is empty. Cannot determine wshape.")
        return Path() # Or handle error appropriately

    # Create a simple namespace for compatibility with PIPNet's get_patch_size
    args = Namespace(
        net=cfg.model.name,
        p_gaussian_hw=cfg.p_gaussian_hw,
        image_size=cfg.dataset.image_size,
        wshape=wshape
    )
    patchsize, skip = get_patch_size(args)
    
    scores_per_prototype = {}
    
    project_iter = tqdm(enumerate(projectloader), total=len(projectloader), desc='Pass 1/2: Collecting scores')
    for i, (xs, _) in project_iter:
        xs = xs.to(device)
        with torch.no_grad():
            output = model(xs)
            pooled_scores = output.proto_fvec.squeeze(0)
            for p in range(num_prototypes):
                if torch.max(classification_weights[:, p]) > 1e-5:
                    scores_per_prototype.setdefault(p, []).append((i, pooled_scores[p].item()))

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(exist_ok=True)
    csv_filepath = output_dir / f"purity_eval_top{cfg.k_top_patches}.csv"
    
    proto_img_coordinates = []
    with open(csv_filepath, "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["prototype", "img name", "h_min_224", "h_max_224", "w_min_224", "w_max_224"])
        
        proto_iter = tqdm(scores_per_prototype.keys(), total=len(scores_per_prototype), desc='Pass 2/2: Writing CSV')
        for prototype in proto_iter:
            df = pd.DataFrame(scores_per_prototype[prototype], columns=['img_id', 'scores'])
            topk = df.nlargest(cfg.k_top_patches, 'scores')
            
            for _, row in topk.iterrows():
                imgid = int(row['img_id'])
                imgname = imgs[imgid][0]
                imgtensor = projectloader.dataset[imgid][0].unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(imgtensor)
                    pfs = output.proto_fmap.squeeze(0)
                    location_h, location_h_idx = torch.max(pfs[prototype, :, :], dim=0)
                    _, location_w_idx = torch.max(location_h, dim=0)
                    
                    h_min, h_max, w_min, w_max = get_img_coordinates(
                        cfg.dataset.image_size, pfs.shape, patchsize, skip,
                        location_h_idx[location_w_idx].item(), location_w_idx.item()
                    )
                    proto_img_coordinates.append([prototype, imgname, h_min, h_max, w_min, w_max])
        
        writer.writerows(proto_img_coordinates)
        
    print(f"\nSuccessfully wrote CSV to {csv_filepath}")
    return csv_filepath


def eval_purity_from_csv(csv_path: Path, cfg: PurityBenchConfig):
    """Evaluates prototype purity from the generated CSV file, with fixes."""
    
    # Dynamically determine wshape from the model config if possible, or keep hardcoded logic
    if 'deit' in cfg.model.name or 'vit' in cfg.model.name: 
        wshape = 14
    else: 
        # For ConvNeXt_tiny_13, wshape is 13. For original PIPNet, it's 7.
        # This part needs to be correct for the specific model being evaluated.
        # Let's assume it's correctly set for the model that generated the CSV.
        # For the official ConvNeXt_tiny_13, this should be 13.
        # For the original PIPNet ResNet/etc, this would be 7.
        # We will assume wshape=7 for the classic PIPNet model evaluation for now.
        wshape = 7
        if '13' in cfg.model.name: # A simple heuristic
             wshape = 13
        
    args = Namespace(
        net=cfg.model.name,
        p_gaussian_hw=cfg.p_gaussian_hw,
        image_size=cfg.dataset.image_size,
        wshape=wshape
    )
    patchsize, _ = get_patch_size(args)
    imgresize = float(cfg.dataset.image_size)
    
    cub_root = Path(cfg.cub_cropped_data_path)
    if (cub_root / "CUB_200_2011").is_dir():
        cub_root = cub_root / "CUB_200_2011"
        
    parts_loc_path = cub_root / "parts/part_locs.txt"
    parts_name_path = cub_root / "parts/parts.txt"
    imgs_id_path = cub_root / "images.txt"

    path_to_id, img_to_part_xy_vis, parts_id_to_name = {}, {}, {}
    with open(imgs_id_path) as f:
        for line in f: id, path = line.strip().split(' '); path_to_id[path] = id
    with open(parts_loc_path) as f:
        for line in f:
            img, partid, x, y, vis = line.strip().split(' ')
            if vis == '1': img_to_part_xy_vis.setdefault(img, {})[partid] = (float(x), float(y))
    with open(parts_name_path) as f:
        for line in f: id, name = line.strip().split(' ', 1); parts_id_to_name[id] = name
        
    # --- FIX #2: LOGIC TO MERGE SYMMETRIC PARTS ---
    parts_name_to_id = {v: k for k, v in parts_id_to_name.items()}
    duplicate_part_ids = []
    for id, name in parts_id_to_name.items():
        if 'left' in name:
            new_name = name.replace('left', 'right')
            if new_name in parts_name_to_id:
                duplicate_part_ids.append((id, parts_name_to_id[new_name]))
    
    print("CUB Parts:", parts_id_to_name)
    proto_parts_presences = {}

    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        next(reader) # Skip header
        for p, imgname, h_min, h_max, w_min, w_max in reader:
            presences = proto_parts_presences.setdefault(p, {})
            
            p_imgname = Path(imgname)
            img_rel_path = (Path(p_imgname.parent.name) / p_imgname.name).as_posix()
            img_id = path_to_id.get(img_rel_path)
            if not img_id or img_id not in img_to_part_xy_vis: continue
            
            correct_img_path = cub_root / "images" / img_rel_path
            with Image.open(correct_img_path) as img: w_orig, h_orig = img.size
                
            h_min, h_max, w_min, w_max = map(float, [h_min, h_max, w_min, w_max])

            # --- FIX #1: APPLY PATCH SIZE CORRECTION ---
            if (h_max - h_min) > patchsize:
                correction = (h_max - h_min) - patchsize
                h_min += correction / 2.
                h_max -= correction / 2.
            if (w_max - w_min) > patchsize:
                correction = (w_max - w_min) - patchsize
                w_min += correction / 2.
                w_max -= correction / 2.
            
            h_min_orig, h_max_orig = (h_orig / imgresize) * h_min, (h_orig / imgresize) * h_max
            w_min_orig, w_max_orig = (w_orig / imgresize) * w_min, (w_orig / imgresize) * w_max
            
            parts_in_img = img_to_part_xy_vis[img_id]
            for partid, (x, y) in parts_in_img.items():
                in_patch = 1 if (w_min_orig <= x <= w_max_orig and h_min_orig <= y <= h_max_orig) else 0
                presences.setdefault(partid, []).append(in_patch)

            # --- APPLY PART MERGING LOGIC ---
            for pair in duplicate_part_ids:
                left_part, right_part = pair
                # Check if the left part was present in this image and its presence was recorded
                if left_part in parts_in_img and left_part in presences:
                    # If right part also present, combine scores
                    if right_part in parts_in_img and right_part in presences:
                        # Get the last appended presence value for both
                        presence_left = presences[left_part][-1]
                        presence_right = presences[right_part][-1]
                        # The merged part (right) takes the max presence of the two
                        presences[right_part][-1] = max(presence_left, presence_right)
                    # If only left part was present, transfer its score to the right part
                    else:
                        presences.setdefault(right_part, []).append(presences[left_part][-1])
                    # Remove the now-redundant left part presence
                    del presences[left_part]

    print("\n--- Purity Evaluation Results ---")
    max_presence_purity = {p: max((np.mean(pres) for pres in parts.values()), default=0.0) for p, parts in proto_parts_presences.items()}
    prototypes_part_related = sum(1 for p in max_presence_purity.values() if p > 0.5)
    mean_purity = np.mean(list(max_presence_purity.values())) if max_presence_purity else 0.0
    std_purity = np.std(list(max_presence_purity.values())) if max_presence_purity else 0.0
    
    print(f"Number of prototypes analyzed: {len(proto_parts_presences)}")
    print(f"Number of part-related prototypes (purity > 0.5): {prototypes_part_related}")
    print(f"Mean purity of prototypes: {mean_purity:.4f}")
    print(f"Std deviation of purity: {std_purity:.4f}")


@hydra.main(config_path=".", config_name="purity_config", version_base="1.2")
def run_benchmark(cfg: PurityBenchConfig):
    """Main function to orchestrate the benchmark, configured by Hydra."""
    print("--- PIP-Net Purity Benchmark ---")
    print(OmegaConf.to_yaml(cfg))

    # Resolve paths to be absolute, making them robust to where the script is run from
    cfg.checkpoint_path = to_absolute_path(cfg.checkpoint_path)
    cfg.cub_cropped_data_path = to_absolute_path(cfg.cub_cropped_data_path)
    cfg.output_dir = to_absolute_path(cfg.output_dir)
    
    if cfg.csv_to_eval:
        # --- EVALUATION-ONLY MODE ---
        print(f"\n--- Running in Evaluation-Only Mode ---")
        csv_filepath = Path(to_absolute_path(cfg.csv_to_eval))
        if not csv_filepath.is_file():
            raise FileNotFoundError(f"The provided CSV file was not found: {csv_filepath}")
        print(f"Evaluating from existing file: {csv_filepath}")
        eval_purity_from_csv(csv_path=csv_filepath, cfg=cfg)
        print("\n--- Benchmark finished successfully! ---")
        return 
    
    # The build_pipnet_model function expects this specific attribute to be set
    cfg.pipnet_checkpoint_path = cfg.checkpoint_path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from: {cfg.checkpoint_path}")
    model, _ = build_pipnet_model(cfg, device) # type: ignore
    model.to(device).eval()

    # Define the deterministic transform pipeline for the benchmark
    benchmark_transform = transforms_v2.Compose([
        transforms_v2.Resize((cfg.dataset.image_size, cfg.dataset.image_size), interpolation=transforms_v2.InterpolationMode.BICUBIC, antialias=True),
        transforms_v2.ToTensor(),
        transforms_v2.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    print(f"Loading dataset from: {cfg.cub_cropped_data_path}")
    _, val_ds = get_dataset(name="cub200", val_transform=benchmark_transform, path=cfg.cub_cropped_data_path, train_transform=benchmark_transform)
    projectloader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)

    csv_filepath = generate_purity_csv(model=model, projectloader=projectloader, cfg=cfg, device=device)
    eval_purity_from_csv(csv_path=csv_filepath, cfg=cfg)
    
    print("\n--- Benchmark finished successfully! ---")


if __name__ == "__main__":
    run_benchmark()
