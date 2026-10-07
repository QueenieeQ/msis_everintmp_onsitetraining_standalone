# training_common.py
# ------------------------------------------------------------
# Helpers shared by training.py (ROI one-shot trainer) and
# training_screw.py (screw one-shot trainer): logging setup, dataset-gen
# config, YOLO11s training, and best-effort directory cleanup. Bbox/affine
# geometry and augmentations live in training_geometry.py (split out
# separately to keep this file, and both trainer files, under 300 lines).
# ------------------------------------------------------------

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO


# ============================================================
# Logging
# ============================================================
def setup_logger(log_dir: Path, name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = RotatingFileHandler(
        log_dir / f"{name}.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


# ============================================================
# Dataset generation config (shared shape; ROI/screw each build their own
# dataset differently, see generate_dataset in training.py/training_screw.py)
# ============================================================
@dataclass
class GenConfig:
    n_images: int
    val_ratio: float
    imgsz: int
    seed: int
    aug_cfg: Dict[str, Any]


def safe_rmtree(path: Path, logger: logging.Logger) -> None:
    try:
        if path.exists():
            shutil.rmtree(path)
            logger.info(f"Deleted: {path}")
    except Exception as e:
        logger.warning(f"Failed to delete {path}: {e}")


# ============================================================
# Label loading (ref_user.txt: "class_id x1 y1 x2 y2" per line)
# ============================================================
def load_labels(label_path: Path, logger: logging.Logger) -> List[Tuple[int, np.ndarray]]:
    """Load labels from ref_user.txt format: class_id x1 y1 x2 y2.

    Shared by training_screw.py (all holes, full image) and
    training_patch.py (one hole at a time, cropped to a fixed patch).
    """
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    labels = []
    with label_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            bbox = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])], dtype=np.float32)
            labels.append((class_id, bbox))

    logger.info(f"Loaded {len(labels)} labels from {label_path}")
    return labels


# ============================================================
# Training
# ============================================================
def train_yolo11s(
    dataset_dir: Path,
    printer_name: str,
    train_cfg: Dict[str, Any],
    logger: logging.Logger,
    ref_img_path: Path,
    run_prefix: str,
    weights_subdir: Optional[Path] = None,
    weights_filename: str = "weight.pt",
) -> Path:
    """Train a YOLO model on dataset_dir/data.yaml and copy best.pt next to
    the reference data. `run_prefix` distinguishes ROI ("roi") vs. screw
    ("screw") vs. patch ("patch") runs.

    `weights_subdir` (relative, optional) nests the copied weights under
    `<ref_root>/weights/<weights_subdir>/` instead of directly in
    `<ref_root>/weights/`; `weights_filename` overrides the default
    "weight.pt" — used by the patch trainer so each hole's weights land at
    `weights/patches_weights/hole_{i}/weight_hole_{i}.pt` without colliding
    with the shared ROI/screw `weights/weight.pt`.
    """
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing data.yaml at: {data_yaml}")

    epochs = int(train_cfg.get("epochs", 60))
    imgsz = int(train_cfg.get("imgsz", 640))
    batch = int(train_cfg.get("batch", 16))
    device = str(train_cfg.get("device", "0"))
    workers = int(train_cfg.get("workers", 4))
    patience = int(train_cfg.get("patience", 12))
    project_dir = str(train_cfg.get("project_dir", "runs"))
    amp = bool(train_cfg.get("amp", False))
    cudnn_enabled = bool(train_cfg.get("cudnn_enabled", False))
    model_variant = str(train_cfg.get("model_variant", "yolo11s.pt"))

    run_name = f"{run_prefix}_{printer_name}"

    torch.backends.cudnn.enabled = cudnn_enabled

    logger.info(f"Starting YOLO training (model_variant={model_variant})...")
    logger.info(f"data={data_yaml}")
    logger.info(
        f"epochs={epochs}, imgsz={imgsz}, batch={batch}, device={device}, "
        f"workers={workers}, patience={patience}, amp={amp}, cudnn_enabled={cudnn_enabled}"
    )

    model = YOLO(model_variant)

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        patience=patience,
        name=run_name,
        project=project_dir,
        exist_ok=True,
        verbose=True,
        amp=amp,
    )

    save_dir = Path(getattr(results, "save_dir", "")) if results is not None else Path(project_dir) / "detect" / run_name
    if not save_dir.exists():
        save_dir = Path(project_dir) / "detect" / run_name

    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        candidates = list(save_dir.rglob("best.pt"))
        if not candidates:
            raise FileNotFoundError(f"Could not find best.pt under: {save_dir}")
        best_pt = candidates[0]

    # -----------------------------------------
    # Save weights next to reference data
    # -----------------------------------------
    # ref image path example:
    # data/inference/save_ref_data/SLP-DX420/images/ref_user.png
    ref_root = ref_img_path.parents[1]   # -> data/inference/save_ref_data/SLP-DX420
    weights_dir = ref_root / "weights"
    if weights_subdir is not None:
        weights_dir = weights_dir / weights_subdir
    weights_dir.mkdir(parents=True, exist_ok=True)

    out_weights = weights_dir / weights_filename
    shutil.copy2(best_pt, out_weights)

    logger.info(f"Copied best weights to: {out_weights}")
    return out_weights
