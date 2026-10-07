# training_geometry.py
# ------------------------------------------------------------
# Bbox/affine geometry and photometric augmentation helpers shared by
# training.py (ROI one-shot trainer) and training_screw.py (screw one-shot
# trainer) — previously byte-identical/near-identical copies in both files.
# ------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np


# ============================================================
# Geometry helpers
# ============================================================
def clamp_bbox_xyxy(b: np.ndarray, w: int, h: int) -> np.ndarray:
    b = b.astype(np.float32)
    b[0] = np.clip(b[0], 0, w - 1)
    b[1] = np.clip(b[1], 0, h - 1)
    b[2] = np.clip(b[2], 0, w - 1)
    b[3] = np.clip(b[3], 0, h - 1)
    if b[2] <= b[0]:
        b[2] = min(w - 1, b[0] + 1)
    if b[3] <= b[1]:
        b[3] = min(h - 1, b[1] + 1)
    return b


def bbox_xyxy_to_yolo(b: np.ndarray, w: int, h: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = b.astype(np.float32)
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return float(cx), float(cy), float(bw), float(bh)


def warp_affine_and_update_bbox(
    img: np.ndarray,
    bbox_xyxy: np.ndarray,
    M: np.ndarray,
    out_w: int,
    out_h: int,
    border_mode=cv2.BORDER_REFLECT_101,
) -> Tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = bbox_xyxy.astype(np.float32)
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    ones = np.ones((4, 1), dtype=np.float32)
    pts = np.hstack([corners, ones])  # (4,3)
    new_pts = (M @ pts.T).T          # (4,2)

    nx1 = np.min(new_pts[:, 0])
    ny1 = np.min(new_pts[:, 1])
    nx2 = np.max(new_pts[:, 0])
    ny2 = np.max(new_pts[:, 1])

    new_bbox = np.array([nx1, ny1, nx2, ny2], dtype=np.float32)
    new_bbox = clamp_bbox_xyxy(new_bbox, out_w, out_h)

    warped = cv2.warpAffine(img, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=border_mode)
    return warped, new_bbox


def bbox_after_affine(bbox_xyxy: np.ndarray, M: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """
    Affine-transform a bbox and clamp, WITHOUT warping any image.

    This is exactly the bbox half of warp_affine_and_update_bbox (same float32
    casts, same operation order, same clamp_bbox_xyxy call), so the returned
    bbox is bit-for-bit identical to what that function returns. It uses no RNG
    and does not depend on image content, so replacing the per-label image warps
    with this call does not change dataset generation or training results.
    """
    x1, y1, x2, y2 = bbox_xyxy.astype(np.float32)
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    ones = np.ones((4, 1), dtype=np.float32)
    pts = np.hstack([corners, ones])
    new_pts = (M @ pts.T).T

    nx1 = np.min(new_pts[:, 0])
    ny1 = np.min(new_pts[:, 1])
    nx2 = np.max(new_pts[:, 0])
    ny2 = np.max(new_pts[:, 1])

    new_bbox = np.array([nx1, ny1, nx2, ny2], dtype=np.float32)
    new_bbox = clamp_bbox_xyxy(new_bbox, out_w, out_h)
    return new_bbox


# ============================================================
# Augmentations (photometric)
# ============================================================
def aug_brightness(img: np.ndarray, rng: np.random.Generator, br_cfg: Dict[str, Any]) -> np.ndarray:
    gain = float(rng.uniform(br_cfg["gain_min"], br_cfg["gain_max"]))
    bias = float(rng.uniform(br_cfg["bias_min"], br_cfg["bias_max"]))
    out = img.astype(np.float32) * gain + bias
    return np.clip(out, 0, 255).astype(np.uint8)


def aug_noise(img: np.ndarray, rng: np.random.Generator, noise_cfg: Dict[str, Any]) -> np.ndarray:
    sigma = float(rng.uniform(noise_cfg["sigma_min"], noise_cfg["sigma_max"]))
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def aug_blur(img: np.ndarray, rng: np.random.Generator, blur_cfg: Dict[str, Any]) -> np.ndarray:
    kernels = blur_cfg.get("kernels", [3, 5, 7])
    k = int(rng.choice(kernels))
    if rng.random() < 0.5:
        return cv2.GaussianBlur(img, (k, k), 0)
    return cv2.medianBlur(img, k)


# ============================================================
# Augmentations (geometric - composed affine)
# ============================================================
def build_affine(
    w: int,
    h: int,
    rng: np.random.Generator,
    do_translate: bool,
    do_rotate: bool,
    do_shear: bool,
    do_scale: bool,
    aug_cfg: Dict[str, Any],
) -> np.ndarray:
    cx, cy = w / 2.0, h / 2.0

    angle = 0.0
    scale = 1.0
    tx = 0.0
    ty = 0.0
    shx = 0.0
    shy = 0.0

    if do_rotate:
        rc = aug_cfg["rotate"]
        angle = float(rng.uniform(rc["deg_min"], rc["deg_max"]))

    if do_scale:
        sc = aug_cfg["scale"]
        scale = float(rng.uniform(sc["min"], sc["max"]))

    if do_translate:
        tc = aug_cfg["translate"]
        tx = float(rng.uniform(-tc["frac_x"], tc["frac_x"]) * w)
        ty = float(rng.uniform(-tc["frac_y"], tc["frac_y"]) * h)

    if do_shear:
        sh = aug_cfg["shear"]
        shx = float(rng.uniform(sh["shx_min"], sh["shx_max"]))
        shy = float(rng.uniform(sh["shy_min"], sh["shy_max"]))

    R = cv2.getRotationMatrix2D((cx, cy), angle, scale)  # 2x3
    S = np.array([[1.0, shx, -shx * cy],
                  [shy, 1.0, -shy * cx]], dtype=np.float32)
    T = np.array([[1.0, 0.0, tx],
                  [0.0, 1.0, ty]], dtype=np.float32)

    def to33(A23: np.ndarray) -> np.ndarray:
        return np.vstack([A23, np.array([0.0, 0.0, 1.0], dtype=np.float32)])

    M33 = to33(T) @ to33(S) @ to33(R)
    return M33[:2, :].astype(np.float32)
