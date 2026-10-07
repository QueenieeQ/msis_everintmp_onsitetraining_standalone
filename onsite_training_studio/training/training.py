# roi_oneshot_trainer.py
# ------------------------------------------------------------
# Robot-facing ONE function with ONLY 3 inputs:
#   train_roi_one_shot(image, bbox, printer_name)
#
# - Reads training/augmentation params from YAML every call (hot reload)
# - Builds a temporary YOLO dataset with N augmented images
# - Trains YOLO11s and saves weights to: ./<printer_name>_roi.pt
# - Deletes augmented dataset after training (always)
# - Adds logging (console + file)
#
# Shared helpers live in training_defaults.py / training_common.py /
# training_geometry.py (also used by training_screw.py).
#
# Requirements:
#   pip install ultralytics opencv-python numpy pyyaml
# ------------------------------------------------------------

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from onsite_training_studio.training.training_defaults import load_yaml_config, onsite_yaml_path
from onsite_training_studio.training.training_common import GenConfig, safe_rmtree, setup_logger, train_yolo11s
from onsite_training_studio.training.training_geometry import (
    aug_blur,
    aug_brightness,
    aug_noise,
    build_affine,
    bbox_xyxy_to_yolo,
    clamp_bbox_xyxy,
    warp_affine_and_update_bbox,
)


# ============================================================
# Dataset generation (ROI: single bbox per image)
# ============================================================
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


def generate_dataset(
    ref_img_path: Path,
    ref_bbox_xyxy: Tuple[int, int, int, int],
    printer_name: str,
    out_root: Path,
    cfg: GenConfig,
    logger: logging.Logger,
) -> Path:
    rng = np.random.default_rng(cfg.seed)

    img = cv2.imread(str(ref_img_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {ref_img_path}")

    h0, w0 = img.shape[:2]
    bbox = clamp_bbox_xyxy(np.array(ref_bbox_xyxy, dtype=np.float32), w0, h0)

    ds_dir = out_root / f"tmp_yolo_roi_{printer_name}_{uuid.uuid4().hex[:8]}"
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
        b = bbox.copy()

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
            base, b = warp_affine_and_update_bbox(base, b, M, w0, h0)

        if do_brightness:
            base = aug_brightness(base, rng, p["brightness"])
        if do_noise:
            base = aug_noise(base, rng, p["noise"])
        if do_blur:
            base = aug_blur(base, rng, p["blur"])

        base, b = letterbox_to(base, b, cfg.imgsz)

        cx, cy, bw, bh = bbox_xyxy_to_yolo(b, cfg.imgsz, cfg.imgsz)
        stem = f"{printer_name}_{split}_{i:04d}"

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
        f"  0: roi\n"
    )
    (ds_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")

    logger.info("Dataset generation complete.")
    return ds_dir


# ============================================================
# PUBLIC ROBOT API (ONLY 3 inputs)
# ============================================================
def train_roi_one_shot(
    image: Union[str, np.ndarray],
    bbox: Union[Tuple[int, int, int, int], Tuple[float, float, float, float]],
    printer_name: str,
    model_variant: Optional[str] = None,
) -> str:
    """
    Robot calls ONLY this with 3 inputs.
    """

    from onsite_training_studio.paths import tmp_dir

    cwd = tmp_dir()
    cwd.mkdir(parents=True, exist_ok=True)
    yaml_path = onsite_yaml_path()
    cfg = load_yaml_config(yaml_path, profile="roi")

    from onsite_training_studio.paths import logs_dir

    logger = setup_logger(
        logs_dir() / "onsite_training",
        name=f"roi_train_{printer_name}",
    )

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

    # ---- load image ----
    tmp_ref_path: Optional[Path] = None
    if isinstance(image, str):
        ref_img_path = Path(image)
        if not ref_img_path.exists():
            raise FileNotFoundError(ref_img_path)
        img = cv2.imread(str(ref_img_path))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {ref_img_path}")
    elif isinstance(image, np.ndarray):
        img = image.copy()
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("image must be BGR np.ndarray with shape (H,W,3).")
        tmp_ref_path = cwd / f"_tmp_ref_{printer_name}_{uuid.uuid4().hex[:6]}.jpg"
        ok = cv2.imwrite(str(tmp_ref_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise RuntimeError(f"Could not write temp reference image: {tmp_ref_path}")
        ref_img_path = tmp_ref_path
    else:
        raise TypeError("image must be a file path (str) or numpy array (np.ndarray).")

    h, w = img.shape[:2]

    # ---- parse bbox ----
    if not (isinstance(bbox, (tuple, list)) and len(bbox) == 4):
        raise ValueError("bbox must be (x1, y1, x2, y2)")

    b = np.array(bbox, dtype=np.float32)
    if np.all(b >= 0.0) and np.all(b <= 1.5):  # normalized xyxy
        b[0] *= w
        b[2] *= w
        b[1] *= h
        b[3] *= h

    x1, y1, x2, y2 = b.tolist()
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    ref_bbox = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))

    logger.info("==============================================")
    logger.info(" ROI ONE-SHOT TRAIN START")
    logger.info("==============================================")
    logger.info(f"Printer name     : {printer_name}")
    logger.info(f"Ref image         : {ref_img_path}")
    logger.info(f"Ref image size    : {w}x{h}")
    logger.info(f"Ref bbox (px)     : {ref_bbox}")
    logger.info(f"Config YAML       : {yaml_path}")
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
            ref_bbox_xyxy=ref_bbox,
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
            run_prefix="roi",
        )

        logger.info(f"✅ FINAL WEIGHTS: {out_weights}")
        return str(out_weights.resolve())

    finally:
        if ds_dir is not None:
            safe_rmtree(ds_dir, logger)

        if tmp_ref_path is not None:
            try:
                if tmp_ref_path.exists():
                    tmp_ref_path.unlink()
            except Exception as e:
                logger.warning(f"Could not delete temp ref image {tmp_ref_path}: {e}")

        if delete_runs_dir:
            run_dir = Path(project_dir) / "detect" / f"roi_{printer_name}"
            safe_rmtree(run_dir, logger)

        logger.info("==============================================")
        logger.info(" ROI ONE-SHOT TRAIN END")
        logger.info("==============================================")


# ============================================================
# Optional local test
# ============================================================
if __name__ == "__main__":
    ref_img = r"C:\Users\sajja\Downloads\New folder (2)\ref_user.png"
    test_bbox = (416, 223, 812, 565)
    printer_name= "SLP-DX420"
    print(train_roi_one_shot(ref_img, test_bbox, printer_name))
