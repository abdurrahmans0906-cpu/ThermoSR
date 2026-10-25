import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
from app.core.logging import logger
from app.services.preprocessing import load_thermal_image, load_rgb_image

def detect_dataset_structure(root_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Detects whether root directory matches CIDIS layout or FLIR Challenge layout.
    """
    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {root_path}")

    # Check for CIDIS layout (train/RGB, train/Thermal)
    is_cidis = False
    for split in ["train", "validation", "test"]:
        if (root / split / "Thermal").exists() or (root / split / "thermal").exists():
            is_cidis = True
            break

    # Check for FLIR Challenge layout (train/640_flir_hr, test/640_flir_hr)
    is_flir_challenge = False
    for split in ["train", "test"]:
        if (root / split / "640_flir_hr").exists():
            is_flir_challenge = True
            break

    if is_cidis:
        return "cidis", {"layout": "cidis", "splits": ["train", "validation", "test"]}
    elif is_flir_challenge:
        return "challenge_flir", {"layout": "challenge_flir", "splits": ["train", "test"]}
    else:
        # Generic folder with subfolders
        return "generic", {"layout": "generic", "splits": [d.name for d in root.iterdir() if d.is_dir()]}

def scan_dataset(root_path: str) -> Dict[str, Any]:
    """
    Scans dataset directory, pairs thermal and RGB images, verifies file readability,
    and returns comprehensive metadata report.
    """
    root = Path(root_path)
    dataset_type, meta = detect_dataset_structure(root_path)

    discovered_pairs = []
    unmatched_files = []
    aspect_warnings = []
    split_counts = {}

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")

    if dataset_type == "cidis":
        for split in ["train", "validation", "test"]:
            t_dir = root / split / "Thermal"
            if not t_dir.exists():
                t_dir = root / split / "thermal"
            rgb_dir = root / split / "RGB"
            if not rgb_dir.exists():
                rgb_dir = root / split / "rgb"

            if not t_dir.exists():
                continue

            thermal_files = []
            for ext in extensions:
                thermal_files.extend(list(t_dir.glob(ext)))
                thermal_files.extend(list(t_dir.glob(ext.upper())))

            thermal_files = sorted(list(set(thermal_files)))
            split_count = 0

            for t_file in thermal_files:
                stem = t_file.stem
                # Find matching RGB file
                matched_rgb = None
                if rgb_dir.exists():
                    for ext in [".jpg", ".png", ".jpeg", ".tif", ".JPG", ".PNG"]:
                        cand = rgb_dir / f"{stem}{ext}"
                        if cand.exists():
                            matched_rgb = cand
                            break

                # Get dimensions
                try:
                    with Image.open(t_file) as im_t:
                        t_w, t_h = im_t.size
                    
                    rgb_w, rgb_h = None, None
                    if matched_rgb:
                        with Image.open(matched_rgb) as im_r:
                            rgb_w, rgb_h = im_r.size
                            if abs((rgb_w / rgb_h) - (t_w / t_h)) > 0.02:
                                aspect_warnings.append({
                                    "file": stem,
                                    "split": split,
                                    "thermal_dims": [t_w, t_h],
                                    "rgb_dims": [rgb_w, rgb_h]
                                })

                    pair_info = {
                        "split": split,
                        "stem": stem,
                        "thermal_path": str(t_file.relative_to(root)),
                        "rgb_path": str(matched_rgb.relative_to(root)) if matched_rgb else None,
                        "dimensions": {
                            "thermal": [t_w, t_h],
                            "rgb": [rgb_w, rgb_h] if matched_rgb else None
                        },
                        "validation_status": "valid" if matched_rgb else "missing_rgb",
                        "alignment_status": "aspect_mismatch" if (matched_rgb and abs((rgb_w / rgb_h) - (t_w / t_h)) > 0.02) else "aligned"
                    }
                    discovered_pairs.append(pair_info)
                    split_count += 1
                except Exception as e:
                    unmatched_files.append({"path": str(t_file), "error": str(e)})

            split_counts[split] = split_count

    elif dataset_type == "challenge_flir":
        for split in ["train", "test"]:
            hr_dir = root / split / "640_flir_hr"
            lr_dir = root / split / "160_domo_lr"
            mr_dir = root / split / "320_axis_mr"

            if not hr_dir.exists():
                continue

            hr_files = []
            for ext in extensions:
                hr_files.extend(list(hr_dir.glob(ext)))
            hr_files = sorted(list(set(hr_files)))

            split_count = 0
            for hr_file in hr_files:
                stem = hr_file.stem
                # Check for corresponding 160 or 320 thermal file
                matched_lr = None
                if lr_dir.exists():
                    cand = lr_dir / f"{stem}.jpg"
                    if cand.exists():
                        matched_lr = cand

                try:
                    with Image.open(hr_file) as im_hr:
                        hr_w, hr_h = im_hr.size
                    
                    pair_info = {
                        "split": split,
                        "stem": stem,
                        "thermal_path": str(hr_file.relative_to(root)),
                        "rgb_path": None,  # Multi-res thermal challenge dataset
                        "reference_path": str(matched_lr.relative_to(root)) if matched_lr else None,
                        "dimensions": {
                            "thermal": [hr_w, hr_h],
                            "rgb": None
                        },
                        "validation_status": "valid",
                        "alignment_status": "thermal_only_paired"
                    }
                    discovered_pairs.append(pair_info)
                    split_count += 1
                except Exception as e:
                    unmatched_files.append({"path": str(hr_file), "error": str(e)})

            split_counts[split] = split_count

    valid_count = sum(1 for p in discovered_pairs if p["validation_status"] == "valid")

    return {
        "dataset_type": dataset_type,
        "root_dir": str(root),
        "total_pairs": len(discovered_pairs),
        "valid_pairs": valid_count,
        "unmatched_count": len(unmatched_files),
        "split_counts": split_counts,
        "aspect_warnings": aspect_warnings,
        "pairs": discovered_pairs,
    }
