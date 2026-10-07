# training_patch_dataset.py
# ------------------------------------------------------------
# Dataset generation for patch_training: crop a fixed patch_size_px x
# patch_size_px window around one screw hole's center (from its full-image
# bbox), re-base that hole's bbox into patch-local coordinates, then
# augment/letterbox exactly like training.py's single-bbox ROI dataset —
# just sourced from the small crop instead of the full reference image.
# Split out of training_patch.py to keep both files under 200 lines.
# ------------------------------------------------------------

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from onsite_training_studio.training.training_common import GenConfig
from onsite_training_studio.training.training_geometry import (
    aug_blur,
    aug_brightness,
    aug_noise,
    build_affine,
    bbox_xyxy_to_yolo,
    clamp_bbox_xyxy,
    warp_affine_and_update_bbox,
)


def compute_patch_window(cx: float, cy: float, patch_size: int, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Fixed patch_size x patch_size window centered on (cx, cy), shifted to
    stay fully inside the image; clamped (patch smaller than requested)
    only if the image itself is smaller than patch_size.
    """
    half = patch_size / 2.0
    x1, y1 = cx - half, cy - half
    x2, y2 = x1 + patch_size, y1 + patch_size

    if x1 < 0:
        x2 -= x1
        x1 = 0.0
    if y1 < 0:
        y2 -= y1
        y1 = 0.0
    if x2 > img_w:
        x1 -= (x2 - img_w)
        x2 = float(img_w)
    if y2 > img_h:
        y1 -= (y2 - img_h)
        y2 = float(img_h)

    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(img_w), x2), min(float(img_h), y2)
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def crop_patch_and_local_bbox(
    img: np.ndarray, hole_bbox_xyxy: np.ndarray, patch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop the fixed-size patch around a hole's center and return
    (patch_img, local_bbox_xyxy) with the hole's bbox re-based/clipped to
    the patch's own coordinate space.
    """
    img_h, img_w = img.shape[:2]
    hx1, hy1, hx2, hy2 = hole_bbox_xyxy.tolist()
    cx, cy = (hx1 + hx2) / 2.0, (hy1 + hy2) / 2.0

    px1, py1, px2, py2 = compute_patch_window(cx, cy, patch_size, img_w, img_h)
    patch_img = img[py1:py2, px1:px2]

    local_bbox = np.array([hx1 - px1, hy1 - py1, hx2 - px1, hy2 - py1], dtype=np.float32)
    local_bbox = clamp_bbox_xyxy(local_bbox, patch_img.shape[1], patch_img.shape[0])
    return patch_img, local_bbox


def letterbox_to(img: np.ndarray, bbox_xyxy: np.ndarray, new_size: int) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    scale = min(new_size / w, new_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x = (new_size - nw) // 2
    pad_y = (new_size - nh) // 2

    out = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
    out[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized

    b = bbox_xyxy.astype(np.float32).copy()
    b[[0, 2]] = b[[0, 2]] * scale + pad_x
    b[[1, 3]] = b[[1, 3]] * scale + pad_y
    b = clamp_bbox_xyxy(b, new_size, new_size)
    return out, b


def generate_patch_dataset(
    patch_img: np.ndarray,
    local_bbox_xyxy: np.ndarray,
    printer_name: str,
    hole_id: int,
    out_root: Path,
    cfg: GenConfig,
    logger: logging.Logger,
) -> Path:
    rng = np.random.default_rng(cfg.seed)

    h0, w0 = patch_img.shape[:2]
    bbox = clamp_bbox_xyxy(local_bbox_xyxy, w0, h0)

    ds_dir = out_root / f"tmp_yolo_patch_{printer_name}_hole{hole_id}_{uuid.uuid4().hex[:8]}"
    for p in ["images/train", "images/val", "labels/train", "labels/val"]:
        (ds_dir / p).mkdir(parents=True, exist_ok=True)

    n_total = int(cfg.n_images)
    n_val = max(1, int(round(n_total * float(cfg.val_ratio))))
    n_train = n_total - n_val

    aug = cfg.aug_cfg

    logger.info(f"Generating patch dataset (hole {hole_id}) at: {ds_dir}")
    logger.info(f"Train/Val split: {n_train}/{n_val} (total={n_total})")

    def write_sample(i: int, split: str):
        base = patch_img.copy()
        b = bbox.copy()

        p = aug
        do_brightness = (rng.random() < float(p["p_brightness"]))
        do_translate = (rng.random() < float(p["p_translate"]))
        do_noise = (rng.random() < float(p["p_noise"]))
        do_blur = (rng.random() < float(p["p_blur"]))
        do_rotate = (rng.random() < float(p["p_rotate"]))
        do_shear = (rng.random() < float(p["p_shear"]))
        do_scale = (rng.random() < float(p["p_scale"]))

        if do_translate or do_rotate or do_shear or do_scale:
            M = build_affine(w0, h0, rng, do_translate, do_rotate, do_shear, do_scale, p)
            base, b = warp_affine_and_update_bbox(base, b, M, w0, h0)

        if do_brightness:
            base = aug_brightness(base, rng, p["brightness"])
        if do_noise:
            base = aug_noise(base, rng, p["noise"])
        if do_blur:
            base = aug_blur(base, rng, p["blur"])

        base, b = letterbox_to(base, b, cfg.imgsz)

        cx, cy, bw, bh = bbox_xyxy_to_yolo(b, cfg.imgsz, cfg.imgsz)
        stem = f"{printer_name}_hole{hole_id}_{split}_{i:04d}"

        img_out = ds_dir / f"images/{split}/{stem}.jpg"
        lbl_out = ds_dir / f"labels/{split}/{stem}.txt"

        ok = cv2.imwrite(str(img_out), base, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            raise RuntimeError(f"Failed to write image: {img_out}")

        lbl_out.write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")

    for i in range(n_train):
        write_sample(i, "train")
    for i in range(n_val):
        write_sample(i, "val")

    data_yaml = (
        f"path: {ds_dir.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: hole\n"
    )
    (ds_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")

    logger.info("Patch dataset generation complete.")
    return ds_dir
