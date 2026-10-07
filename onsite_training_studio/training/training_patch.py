# training_patch.py
# ------------------------------------------------------------
# patch_training: one YOLO model PER screw hole, trained only on a fixed
# patch_size_px x patch_size_px crop around that hole's center (not the
# full reference image) — unlike train_screw_one_shot, which trains one
# model on all holes together in the full image.
#
# Public API: train_patch_one_shot(printer_name, ref_data_root) -> the
# {hole_id: weights_path} produced, one entry per hole that trained
# successfully (a failed hole is logged and skipped, not fatal to the rest).
#
# Reads the same images/ref_user.png + labels/ref_user.txt that
# train_screw_one_shot reads. Weights land under
# weights/patches_weights/hole_{i}/weight_hole_{i}.pt — parallel to, not
# overwriting, the shared weights/weight.pt used by ROI/screw.
#
# Dataset generation lives in training_patch_dataset.py (kept separate to
# stay under 200 lines here).
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import cv2

from onsite_training_studio.training.training_defaults import load_yaml_config, onsite_yaml_path
from onsite_training_studio.training.training_common import GenConfig, load_labels, safe_rmtree, setup_logger, train_yolo11s
from onsite_training_studio.training.training_patch_dataset import crop_patch_and_local_bbox, generate_patch_dataset


def train_patch_one_shot(
    printer_name: str,
    ref_data_root: str = "data/inference/save_ref_data",
    model_variant: Optional[str] = None,
) -> Dict[int, str]:
    """
    Train one patch-based screw-hole detector per hole.

    Args:
        printer_name: Name of printer/shot (e.g., "SLP-DX220")
        ref_data_root: Root directory containing reference data

    Returns:
        {hole_id: path to that hole's trained weights} for every hole that
        trained successfully.
    """
    from onsite_training_studio.paths import tmp_dir

    cwd = tmp_dir()
    cwd.mkdir(parents=True, exist_ok=True)
    yaml_path = onsite_yaml_path()
    cfg = load_yaml_config(yaml_path, profile="patch")

    from onsite_training_studio.paths import logs_dir

    logger = setup_logger(
        logs_dir() / "onsite_training",
        name=f"patch_train_{printer_name}",
    )

    ref_root = Path(ref_data_root) / printer_name
    ref_img_path = ref_root / "images" / "ref_user.png"
    ref_label_path = ref_root / "labels" / "ref_user.txt"

    img = cv2.imread(str(ref_img_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {ref_img_path}")

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
    patch_size_px = int(ds_cfg.get("patch_size_px", 40))

    tr_cfg.setdefault("imgsz", imgsz)

    delete_runs_dir = bool(tr_cfg.get("delete_runs_dir", False))
    project_dir = str(tr_cfg.get("project_dir", "runs"))

    logger.info("==============================================")
    logger.info(" PATCH ONE-SHOT TRAIN START")
    logger.info("==============================================")
    logger.info(f"Printer name     : {printer_name}")
    logger.info(f"Ref image        : {ref_img_path}")
    logger.info(f"Ref labels       : {ref_label_path}")
    logger.info(f"Num holes        : {len(ref_labels)}")
    logger.info(f"Patch size (px)  : {patch_size_px}")
    logger.info(f"Config YAML      : {yaml_path}")
    logger.info("==============================================")

    gen_cfg = GenConfig(
        n_images=n_images,
        val_ratio=val_ratio,
        imgsz=imgsz,
        seed=seed,
        aug_cfg=aug_cfg,
    )

    results: Dict[int, str] = {}
    last_error: Optional[Exception] = None
    for hole_id, (_class_id, hole_bbox) in enumerate(ref_labels):
        ds_dir: Optional[Path] = None
        try:
            patch_img, local_bbox = crop_patch_and_local_bbox(img, hole_bbox, patch_size_px)

            ds_dir = generate_patch_dataset(
                patch_img=patch_img,
                local_bbox_xyxy=local_bbox,
                printer_name=printer_name,
                hole_id=hole_id,
                out_root=cwd,
                cfg=gen_cfg,
                logger=logger,
            )

            out_weights = train_yolo11s(
                dataset_dir=ds_dir,
                printer_name=f"{printer_name}_hole{hole_id}",
                train_cfg=tr_cfg,
                logger=logger,
                ref_img_path=ref_img_path,
                run_prefix="patch",
                weights_subdir=Path("patches_weights") / f"hole_{hole_id}",
                weights_filename=f"weight_hole_{hole_id}.pt",
            )

            results[hole_id] = str(out_weights.resolve())
            logger.info(f"✅ Hole {hole_id} weights: {out_weights}")

        except Exception as exc:
            logger.error(f"Patch training failed for hole {hole_id}: {exc}")
            last_error = exc

        finally:
            if ds_dir is not None:
                safe_rmtree(ds_dir, logger)
            if delete_runs_dir:
                run_dir = Path(project_dir) / "detect" / f"patch_{printer_name}_hole{hole_id}"
                safe_rmtree(run_dir, logger)

    logger.info(f"Patch training finished: {len(results)}/{len(ref_labels)} holes succeeded")
    if ref_labels and not results:
        # Per-hole failures are tolerated, but total failure must not look like success.
        raise RuntimeError(f"patch training failed for every hole: {last_error}")
    logger.info("==============================================")
    logger.info(" PATCH ONE-SHOT TRAIN END")
    logger.info("==============================================")
    return results


# ============================================================
# Optional local test
# ============================================================
if __name__ == "__main__":
    printer_name = "SLP-DX220"
    print(train_patch_one_shot(printer_name))
