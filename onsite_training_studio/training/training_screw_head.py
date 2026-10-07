# training_screw_head.py
# ------------------------------------------------------------
# One-shot trainer for the SCREW-HEAD (label "S") detector. Separate from the
# screw-hole (H) trainer so each model has its own dataset, run and weights:
#
#   reads : <ref_data_root>/<shot>/images/ref_user.png
#           <ref_data_root>/<shot>/labels/ref_user_screw_head.txt  (class_id x1 y1 x2 y2)
#   writes: <ref_data_root>/<shot>/weights/screw_head/screw_head_detection.pt
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Optional

from onsite_training_studio.training.screw_head import (
    SCREW_HEAD_CLASS_NAME,
    SCREW_HEAD_LABELS_FILE,
    SCREW_HEAD_WEIGHTS_FILENAME,
    SCREW_HEAD_WEIGHTS_SUBDIR,
)
from onsite_training_studio.training.training_common import (
    GenConfig,
    load_labels,
    safe_rmtree,
    setup_logger,
    train_yolo11s,
)
from onsite_training_studio.training.training_defaults import load_yaml_config, onsite_yaml_path
from onsite_training_studio.training.training_screw import generate_dataset


def train_screw_head_one_shot(
    printer_name: str, ref_data_root: str, model_variant: Optional[str] = None
) -> str:
    """Train the screw-head model for one shot; returns the weights path.

    ``printer_name`` is the shot name (same convention as train_screw_one_shot).
    """
    from onsite_training_studio.paths import logs_dir

    from onsite_training_studio.paths import tmp_dir

    cwd = tmp_dir()
    cwd.mkdir(parents=True, exist_ok=True)
    yaml_path = onsite_yaml_path()
    cfg = load_yaml_config(yaml_path, profile="screw")
    logger = setup_logger(
        logs_dir() / "onsite_training",
        name=f"screw_head_train_{printer_name}",
    )

    ref_root = Path(ref_data_root) / printer_name
    ref_img_path = ref_root / "images" / "ref_user.png"
    ref_labels = load_labels(ref_root / "labels" / SCREW_HEAD_LABELS_FILE, logger)

    ds_cfg = cfg["dataset"]
    tr_cfg = dict(cfg["train"])
    if model_variant:
        tr_cfg["model_variant"] = model_variant
    imgsz = int(ds_cfg.get("imgsz", 640))
    tr_cfg.setdefault("imgsz", imgsz)
    gen_cfg = GenConfig(
        n_images=int(ds_cfg.get("n_images", 500)),
        val_ratio=float(ds_cfg.get("val_ratio", 0.10)),
        imgsz=imgsz,
        seed=int(ds_cfg.get("seed", 123)),
        aug_cfg=cfg["augment"],
    )

    logger.info(f"SCREW-HEAD TRAIN START shot={printer_name} objects={len(ref_labels)}")
    ds_dir: Optional[Path] = None
    try:
        ds_dir = generate_dataset(
            ref_img_path=ref_img_path,
            ref_labels=ref_labels,
            printer_name=printer_name,
            out_root=cwd,
            cfg=gen_cfg,
            logger=logger,
            class_name=SCREW_HEAD_CLASS_NAME,
        )
        out_weights = train_yolo11s(
            dataset_dir=ds_dir,
            printer_name=printer_name,
            train_cfg=tr_cfg,
            logger=logger,
            ref_img_path=ref_img_path,
            run_prefix=SCREW_HEAD_CLASS_NAME,
            weights_subdir=Path(SCREW_HEAD_WEIGHTS_SUBDIR),
            weights_filename=SCREW_HEAD_WEIGHTS_FILENAME,
        )
        logger.info(f"FINAL SCREW-HEAD WEIGHTS: {out_weights}")
        return str(out_weights.resolve())
    finally:
        if ds_dir is not None:
            safe_rmtree(ds_dir, logger)
        if bool(tr_cfg.get("delete_runs_dir", False)):
            safe_rmtree(
                Path(str(tr_cfg.get("project_dir", "runs"))) / "detect" / f"{SCREW_HEAD_CLASS_NAME}_{printer_name}",
                logger,
            )
        logger.info("SCREW-HEAD TRAIN END")
