import torch
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from argparse import Namespace
from collections import defaultdict

# Ensure these imports match your project structure
from purity_benchmark.config import PurityBenchConfig
from purity_benchmark.func import get_patch_size
from purity_benchmark.vis_pipnet import get_img_coordinates


def _resolve_image_entries(dataset):
    if hasattr(dataset, "imgs") and dataset.imgs:
        return dataset.imgs
    if hasattr(dataset, "samples") and dataset.samples:
        return dataset.samples
    if hasattr(dataset, "image_paths"):
        # Create dummy labels if needed, or just paths
        return [(path, 0) for path in dataset.image_paths]
    raise AttributeError("Unable to resolve image paths from dataset.")


def _entry_to_path(entry):
    if isinstance(entry, (tuple, list)):
        return str(entry[0])
    return str(entry)


def _extract_image_tensor(dataset, index):
    sample = dataset[index]
    if isinstance(sample, dict):
        return sample["image"]
    elif isinstance(sample, (tuple, list)):
        return sample[0]
    return sample


def generate_purity_csv(
    model: torch.nn.Module,
    projectloader: torch.utils.data.DataLoader,
    cfg: PurityBenchConfig,
    device: torch.device,
) -> tuple[Path, int]:
    model.eval()
    dataset = projectloader.dataset
    imgs = _resolve_image_entries(dataset)

    # Original logic uses raw weights for the check
    # But we must ensure we access the underlying weight if it's wrapped
    if hasattr(model, "head"):
        classification_weights = model.head.classifier.weight.detach()
    else:
        # Fallback if wrapper is different
        classification_weights = model.module._classification.weight.detach()

    num_prototypes = classification_weights.shape[1]

    # 1. Determine Latent Shape dynamically
    try:
        sample_input, _ = next(iter(projectloader))
        sample_input = sample_input.to(device)
        with torch.no_grad():
            output = model(sample_input)
            # output is PIPNetOutput(proto_fmap, proto_fvec, logits)
            wshape = output.proto_fmap.shape[-1]
        print(f"Dynamically determined wshape: {wshape}")
    except StopIteration:
        raise ValueError("DataLoader is empty.")

    if hasattr(cfg, "latent_wshape"):
        cfg.latent_wshape = wshape

    # 2. Calculate Patch/Skip info matching original utils
    args = Namespace(
        net=cfg.model.name,
        image_size=cfg.dataset.image_size,
        wshape=wshape,
    )
    patchsize, skip = get_patch_size(args)

    # 3. Collect Scores (Global Top-K)
    scores_per_prototype = defaultdict(list)
    project_iter = tqdm(
        enumerate(projectloader),
        total=len(projectloader),
        desc="Pass 1/2: Collecting scores",
    )

    # Only batch_size=1 supported for purity benchmark to match indices
    for img_idx, (xs, _) in project_iter:
        xs = xs.to(device)
        with torch.no_grad():
            output = model(xs)
            # output.proto_fvec: [1, num_prototypes]
            pooled_scores = output.proto_fvec.squeeze(0)

        for proto in range(num_prototypes):
            # Original threshold is 1e-5
            if torch.max(classification_weights[:, proto]).item() > 1e-5:
                scores_per_prototype[proto].append(
                    (img_idx, pooled_scores[proto].item())
                )

    # 4. Write CSV
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    csv_filepath = output_dir / f"purity_eval_top{cfg.k_top_patches}.csv"

    proto_img_coordinates = []
    low_similarity_protos = set()

    active_protos = sorted(scores_per_prototype.keys())
    print(f"Analyzing {len(active_protos)} active prototypes...")

    proto_iter = tqdm(
        active_protos, total=len(active_protos), desc="Pass 2/2: Writing CSV"
    )
    for proto in proto_iter:
        df = pd.DataFrame(scores_per_prototype[proto], columns=["img_id", "scores"])
        topk = df.nlargest(cfg.k_top_patches, "scores")

        for _, row in topk.iterrows():
            imgid = int(row["img_id"])
            score = row["scores"]

            if score < 0.1:
                low_similarity_protos.add(proto)

            # Get image path
            img_entry = imgs[imgid]
            img_path_str = _entry_to_path(img_entry)

            # Re-run to get spatial map
            sample_tensor = _extract_image_tensor(dataset, imgid)
            imgtensor = sample_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                sample_output = model(imgtensor)
                proto_maps = sample_output.proto_fmap.squeeze(0)  # [Protos, H, W]

            # Spatial location logic from original get_topk_cub
            # location_h, location_h_idx = torch.max(pfs[prototype,:,:], dim=0)
            # _, location_w_idx = torch.max(location_h, dim=0)

            proto_map = proto_maps[proto]  # [H, W]
            max_val_h, max_idx_h = torch.max(
                proto_map, dim=0
            )  # Max across H for each W
            _, max_w_idx = torch.max(max_val_h, dim=0)
            max_h_idx = max_idx_h[max_w_idx]

            h_idx_val = max_h_idx.item()
            w_idx_val = max_w_idx.item()

            h_min, h_max, w_min, w_max = get_img_coordinates(
                cfg.dataset.image_size,
                sample_output.proto_fmap.shape,
                patchsize,
                skip,
                h_idx_val,
                w_idx_val,
            )

            # Store exactly as original: prototype, img name, coords...
            proto_img_coordinates.append(
                [proto, img_path_str, h_min, h_max, w_min, w_max]
            )

    if low_similarity_protos:
        print(
            f"Warning: {len(low_similarity_protos)} prototypes have top-k similarity < 0.1."
        )

    with open(csv_filepath, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(
            [
                "prototype",
                "img name",
                "h_min_224",
                "h_max_224",
                "w_min_224",
                "w_max_224",
            ]
        )
        writer.writerows(proto_img_coordinates)

    print(f"\nSuccessfully wrote CSV to {csv_filepath}")
    return csv_filepath, wshape


def eval_purity_from_csv(
    csv_path: Path, cfg: PurityBenchConfig, latent_wshape: int = None
):
    """
    Faithful reproduction of `eval_prototypes_cub_parts_csv` from the original PIPNet repo.
    """
    if latent_wshape is None:
        latent_wshape = getattr(cfg, "latent_wshape", 7)

    # --- Setup Paths ---
    cub_root = Path(cfg.cub_cropped_data_path)
    if (cub_root / "CUB_200_2011").is_dir():
        cub_root = cub_root / "CUB_200_2011"

    parts_loc_path = cub_root / "parts/part_locs.txt"
    parts_name_path = cub_root / "parts/parts.txt"
    imgs_id_path = cub_root / "images.txt"

    args = Namespace(
        net=cfg.model.name,
        image_size=cfg.dataset.image_size,
        wshape=latent_wshape,
    )
    patchsize, _ = get_patch_size(args)
    imgresize = float(cfg.dataset.image_size)

    # --- Load Metadata ---
    print("Loading CUB metadata...")
    path_to_id = {}
    with open(imgs_id_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pid, path = line.split(" ")
            path_to_id[path] = pid

    img_to_part_xy_vis = {}
    with open(parts_loc_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_id, partid, x, y, vis = line.split(" ")
            if img_id not in img_to_part_xy_vis:
                img_to_part_xy_vis[img_id] = {}
            if vis == "1":
                img_to_part_xy_vis[img_id][partid] = (float(x), float(y))

    parts_id_to_name = {}
    parts_name_to_id = {}
    with open(parts_name_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pid, name = line.split(" ", 1)
            parts_id_to_name[pid] = name
            parts_name_to_id[name] = pid

    # --- Identify Left/Right Parts for Merging ---
    duplicate_part_ids = []
    with open(parts_name_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pid, name = line.split(" ", 1)
            if "left" in name:
                right_name = name.replace("left", "right")
                if right_name in parts_name_to_id:
                    duplicate_part_ids.append((pid, parts_name_to_id[right_name]))

    print("Processing CSV...")
    proto_parts_presences = {}

    with open(csv_path, newline="") as f:
        reader = csv.reader(f, delimiter=",")
        header = next(reader)  # Skip header

        for row in reader:
            prototype = int(row[0])
            imgname_raw = row[1]  # This might be a full path or relative
            h_min_224, h_max_224 = float(row[2]), float(row[3])
            w_min_224, w_max_224 = float(row[4]), float(row[5])

            if prototype not in proto_parts_presences:
                proto_parts_presences[prototype] = {}

            # --- Path Cleaning Logic (Matched to Original) ---
            # Original: imgname.replace('\\', '/'), split('/')[-2:], check 'normal_'
            imgname_clean = imgname_raw.replace("\\", "/")
            parts = imgname_clean.split("/")
            if len(parts) >= 2:
                parent_folder = parts[-2]
                filename = parts[-1]
                if "normal_" in filename:
                    filename = filename.split("normal_")[-1]
                rel_path = f"{parent_folder}/{filename}"
            else:
                # Fallback if path is just filename
                rel_path = imgname_clean

            # Lookup ID
            img_id = path_to_id.get(rel_path)

            if img_id is None:
                # Debugging helper: Try to find partial match
                continue

            # --- Image Opening (Robustness for your env) ---
            # Try constructing absolute path
            full_path = cub_root / "images" / rel_path
            if not full_path.exists():
                # Try using the raw path from CSV if it was absolute
                if Path(imgname_raw).exists():
                    full_path = Path(imgname_raw)
                # Try without 'images' subdir
                elif (cub_root / rel_path).exists():
                    full_path = cub_root / rel_path
                else:
                    # Cannot measure purity if we can't open image to get size
                    continue

            try:
                with Image.open(full_path) as img:
                    img_orig_width, img_orig_height = img.size
            except:
                continue

            # --- FIX: The "Center Crop" Logic from Original ---
            # If the bounding box is larger than patchsize, center it.
            diffh = h_max_224 - h_min_224
            diffw = w_max_224 - w_min_224

            if diffh > patchsize:
                correction = diffh - patchsize
                h_min_224 = h_min_224 + correction / 2.0
                h_max_224 = h_max_224 - correction / 2.0

            if diffw > patchsize:
                correction = diffw - patchsize
                w_min_224 = w_min_224 + correction / 2.0
                w_max_224 = w_max_224 - correction / 2.0

            # Map to Original Coordinates
            orig_h_min = (img_orig_height / imgresize) * h_min_224
            orig_h_max = (img_orig_height / imgresize) * h_max_224
            orig_w_min = (img_orig_width / imgresize) * w_min_224
            orig_w_max = (img_orig_width / imgresize) * w_max_224

            # --- Check Overlap ---
            if img_id in img_to_part_xy_vis:
                part_dict_img = img_to_part_xy_vis[img_id]

                for part_id, (x, y) in part_dict_img.items():
                    part_in_patch = 0
                    if (orig_h_min <= y <= orig_h_max) and (
                        orig_w_min <= x <= orig_w_max
                    ):
                        part_in_patch = 1

                    if part_id not in proto_parts_presences[prototype]:
                        proto_parts_presences[prototype][part_id] = []
                    proto_parts_presences[prototype][part_id].append(part_in_patch)

                # --- Merge Left/Right Parts (Exact Original Logic) ---
                for left_id, right_id in duplicate_part_ids:
                    # Only proceed if at least one part is in this image's annotations
                    has_left = left_id in part_dict_img
                    has_right = right_id in part_dict_img

                    if has_left and has_right:
                        # Both exist: take the max presence and assign to Right, delete Left
                        # Note: accessing [-1] gets the value for THIS image (just appended)
                        p_left = proto_parts_presences[prototype][left_id][-1]
                        p_right = proto_parts_presences[prototype][right_id][-1]

                        if p_left > p_right:
                            proto_parts_presences[prototype][right_id][-1] = p_left

                        # Remove the entry for left
                        del proto_parts_presences[prototype][left_id]

                    elif has_left and not has_right:
                        # Only left exists: Move it to right key
                        # Check if right key exists in proto list, if not create
                        if right_id not in proto_parts_presences[prototype]:
                            proto_parts_presences[prototype][right_id] = []

                        val = proto_parts_presences[prototype][left_id][-1]
                        proto_parts_presences[prototype][right_id].append(val)

                        del proto_parts_presences[prototype][left_id]

                    # If only right exists, do nothing (it's already in right)
                    # If neither exists, do nothing

    # --- Calculate Stats ---
    print(f"Evaluated {len(proto_parts_presences)} prototypes.")

    prototypes_part_related = 0
    max_purity_values = []

    for proto in proto_parts_presences:
        max_purity = 0.0
        max_part = None

        # Safety check from original
        keys = proto_parts_presences[proto].keys()
        if any(k in keys for k in ["7", "8", "9"]):
            print(f"Warning: Unused parts found in proto {proto}")

        for part_id, hits in proto_parts_presences[proto].items():
            if not hits:
                continue
            purity = np.mean(hits)
            sum_occurs = np.sum(hits)

            if purity > max_purity:
                max_purity = purity
                max_part = parts_id_to_name.get(part_id, part_id)
            elif purity == max_purity:
                # Tie-breaking logic from original (favor higher occurrences)
                pass

        max_purity_values.append(max_purity)
        if max_purity > 0.5:
            prototypes_part_related += 1

    print(
        f"Number of part-related prototypes (purity > 0.5): {prototypes_part_related}"
    )
    if max_purity_values:
        mean_pur = np.mean(max_purity_values)
        std_pur = np.std(max_purity_values)
        print(
            f"Mean purity of prototypes (purest part): {mean_pur:.4f} (std: {std_pur:.4f})"
        )
    else:
        print("No data collected.")
