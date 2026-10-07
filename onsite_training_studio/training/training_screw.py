# roi_oneshot_trainer.py - Modified for Screw Training
# ------------------------------------------------------------
# Robot-facing ONE function with ONLY 2 inputs:
#   train_screw_one_shot(ref_img_path, printer_name)
#
# - Reads reference image from: data/inference/save_ref_data/{printer_name}/images/ref_user.png
# - Reads labels from: data/inference/save_ref_data/{printer_name}/labels/ref_user.txt
# - Label format: class_id x1 y1 x2 y2 (0=screw)
# - Builds a temporary YOLO dataset with N augmented images
# - Trains YOLO11s and saves weights to: data/inference/save_ref_data/{printer_name}/weights/screw_weights.pt
#
# Shared helpers live in training_defaults.py / training_common.py /
# training_geometry.py (also used by training.py).
# ------------------------------------------------------------

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from onsite_training_studio.training.training_defaults import load_yaml_config, onsite_yaml_path
from onsite_training_studio.training.training_common import GenConfig, load_labels, safe_rmtree, setup_logger, train_yolo11s
from onsite_training_studio.training.training_geometry import (
    aug_blur,
    aug_brightness,
    aug_noise,
    bbox_after_affine,
    build_affine,
    bbox_xyxy_to_yolo,
    clamp_bbox_xyxy,
    warp_affine_and_update_bbox,
)


def letterbox_to(img: np.ndarray, bboxes_xyxy: List[np.ndarray], new_size: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    h, w = img.shape[:2]
    scale = min(new_size / w, new_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x = (new_size - nw) // 2
    pad_y = (new_size - nh) // 2

    out = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
    out[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized

    new_bboxes = []
    for bbox in bboxes_xyxy:
        b = bbox.astype(np.float32).copy()
        b[[0, 2]] = b[[0, 2]] * scale + pad_x
        b[[1, 3]] = b[[1, 3]] * scale + pad_y
        b = clamp_bbox_xyxy(b, new_size, new_size)
        new_bboxes.append(b)

    return out, new_bboxes


# Dataset generation (screw: N objects per image)
def generate_dataset(
    ref_img_path: Path,
    ref_labels: List[Tuple[int, np.ndarray]],
    printer_name: str,
    out_root: Path,
    cfg: GenConfig,
    logger: logging.Logger,
    class_name: str = "screw",
) -> Path:
    rng = np.random.default_rng(cfg.seed)

    img = cv2.imread(str(ref_img_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {ref_img_path}")

    h0, w0 = img.shape[:2]

    # Clamp all bboxes
    clamped_labels = []
    for class_id, bbox in ref_labels:
        clamped_bbox = clamp_bbox_xyxy(bbox, w0, h0)
        clamped_labels.append((class_id, clamped_bbox))

    ds_dir = out_root / f"tmp_yolo_{class_name}_{printer_name}_{uuid.uuid4().hex[:8]}"
    for p in ["images/train", "images/val", "labels/train", "labels/val"]:
        (ds_dir / p).mkdir(parents=True, exist_ok=True)

    n_total = int(cfg.n_images)
    n_val = max(1, int(round(n_total * float(cfg.val_ratio))))
    n_train = n_total - n_val

    aug = cfg.aug_cfg

    logger.info(f"Generating dataset at: {ds_dir}")
    logger.info(f"Train/Val split: {n_train}/{n_val} (total={n_total})")

    def write_sample(i: int, split: str):
        base = img.copy()
        current_labels = [(cls, bbox.copy()) for cls, bbox in clamped_labels]

        p = aug
        do_brightness = (rng.random() < float(p["p_brightness"]))
        do_translate  = (rng.random() < float(p["p_translate"]))
        do_noise      = (rng.random() < float(p["p_noise"]))
        do_blur       = (rng.random() < float(p["p_blur"]))
        do_rotate     = (rng.random() < float(p["p_rotate"]))
        do_shear      = (rng.random() < float(p["p_shear"]))
        do_scale      = (rng.random() < float(p["p_scale"]))

        if do_translate or do_rotate or do_shear or do_scale:
            M = build_affine(w0, h0, rng, do_translate, do_rotate, do_shear, do_scale, p)
            base, _ = warp_affine_and_update_bbox(base, current_labels[0][1], M, w0, h0)

            # Update all bboxes with affine transform (bbox-only: image is warped once above)
            current_labels = [(cls, bbox_after_affine(bbox, M, w0, h0)) for cls, bbox in current_labels]

        if do_brightness:
            base = aug_brightness(base, rng, p["brightness"])
        if do_noise:
            base = aug_noise(base, rng, p["noise"])
        if do_blur:
            base = aug_blur(base, rng, p["blur"])

        # Letterbox with all bboxes
        bboxes_only = [bbox for _, bbox in current_labels]
        base, new_bboxes = letterbox_to(base, bboxes_only, cfg.imgsz)

        # Reconstruct labels with class IDs
        final_labels = [(current_labels[i][0], new_bboxes[i]) for i in range(len(current_labels))]

        stem = f"{printer_name}_{split}_{i:04d}"
        img_out = ds_dir / f"images/{split}/{stem}.jpg"
        lbl_out = ds_dir / f"labels/{split}/{stem}.txt"

        ok = cv2.imwrite(str(img_out), base, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            raise RuntimeError(f"Failed to write image: {img_out}")

        # Write all labels
        with lbl_out.open("w") as f:
            for class_id, bbox in final_labels:
                cx, cy, bw, bh = bbox_xyxy_to_yolo(bbox, cfg.imgsz, cfg.imgsz)
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    for i in range(n_train):
        write_sample(i, "train")
    for i in range(n_val):
        write_sample(i, "val")

    data_yaml = (
        f"path: {ds_dir.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: {class_name}\n"
    )
    (ds_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")

    logger.info("Dataset generation complete.")
    return ds_dir


# PUBLIC API
def train_screw_one_shot(
    printer_name: str,
    ref_data_root: str = "data/inference/save_ref_data",
    model_variant: Optional[str] = None,
) -> str:
    """
    Train screw detection model using reference image and labels.

    Args:
        printer_name: Name of printer (e.g., "SLP-DX220")
        ref_data_root: Root directory containing reference data

    Returns:
        Path to trained weights file
    """
    from onsite_training_studio.paths import tmp_dir

    cwd = tmp_dir()
    cwd.mkdir(parents=True, exist_ok=True)
    yaml_path = onsite_yaml_path()
    cfg = load_yaml_config(yaml_path, profile="screw")

    from onsite_training_studio.paths import logs_dir

    logger = setup_logger(
        logs_dir() / "onsite_training",
        name=f"screw_train_{printer_name}",
    )

    # Construct paths
    ref_root = Path(ref_data_root) / printer_name
    ref_img_path = ref_root / "images" / "ref_user.png"
    ref_label_path = ref_root / "labels" / "ref_user.txt"

    # Load labels
    ref_labels = load_labels(ref_label_path, logger)

    ds_cfg = cfg["dataset"]
    tr_cfg = dict(cfg["train"])
    if model_variant:
        tr_cfg["model_variant"] = model_variant
    aug_cfg = cfg["augment"]

    n_images = int(ds_cfg.get("n_images", 500))
    val_ratio = float(ds_cfg.get("val_ratio", 0.10))
    imgsz = int(ds_cfg.get("imgsz", 640))
    seed = int(ds_cfg.get("seed", 123))

    tr_cfg.setdefault("imgsz", imgsz)

    delete_runs_dir = bool(tr_cfg.get("delete_runs_dir", False))
    project_dir = str(tr_cfg.get("project_dir", "runs"))

    logger.info("==============================================")
    logger.info(" SCREW ONE-SHOT TRAIN START")
    logger.info("==============================================")
    logger.info(f"Printer name     : {printer_name}")
    logger.info(f"Ref image        : {ref_img_path}")
    logger.info(f"Ref labels       : {ref_label_path}")
    logger.info(f"Num objects      : {len(ref_labels)}")
    logger.info(f"Config YAML      : {yaml_path}")
    logger.info(f"Dataset: n={n_images}, val_ratio={val_ratio}, imgsz={imgsz}, seed={seed}")
    logger.info(f"Train  : epochs={tr_cfg.get('epochs')}, batch={tr_cfg.get('batch')}, device={tr_cfg.get('device')}")
    logger.info("==============================================")

    gen_cfg = GenConfig(
        n_images=n_images,
        val_ratio=val_ratio,
        imgsz=imgsz,
        seed=seed,
        aug_cfg=aug_cfg,
    )

    ds_dir: Optional[Path] = None
    try:
        ds_dir = generate_dataset(
            ref_img_path=ref_img_path,
            ref_labels=ref_labels,
            printer_name=printer_name,
            out_root=cwd,
            cfg=gen_cfg,
            logger=logger,
        )

        out_weights = train_yolo11s(
            dataset_dir=ds_dir,
            printer_name=printer_name,
            train_cfg=tr_cfg,
            logger=logger,
            ref_img_path=ref_img_path,
            run_prefix="screw",
        )

        logger.info(f"✅ FINAL WEIGHTS: {out_weights}")
        return str(out_weights.resolve())

    finally:
        if ds_dir is not None:
            safe_rmtree(ds_dir, logger)

        if delete_runs_dir:
            run_dir = Path(project_dir) / "detect" / f"screw_{printer_name}"
            safe_rmtree(run_dir, logger)

        logger.info("==============================================")
        logger.info(" SCREW ONE-SHOT TRAIN END")
        logger.info("==============================================")


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    printer_name = "SLP-DX220"
    print(train_screw_one_shot(printer_name))
