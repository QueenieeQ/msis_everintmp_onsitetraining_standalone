"""Runs one onsite-training call for the studio (never raises)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from onsite_training_studio.errors import error_code_str
from onsite_training_studio.training.information_path import (
    OnsiteRegistrationError,
    parse_information_path,
    resolve_registration_yaml,
)
from onsite_training_studio.training.model import build_onsite_training_data

_ERROR_GROUP = "onsite_registration"


def run_training(
    information_path: str,
    run_training: bool,
    output_folder: Optional[str],
    shot: Optional[str] = None,
    model: Optional[str] = None,
    model_variant: Optional[str] = None,
):
    """Returns (ok, comment, output_root, error_code) — never raises.

    `shot`: build/train only this one shot (e.g. "left") independently of
    the others — no need for the other shots' images, and their already-built
    output/weights are left untouched. None (default) builds every shot.

    `model`: train only this model kind (screw_hole|patches|screw_head); the
    shot's other weights are kept. `model_variant`: YOLO checkpoint (e.g.
    yolo11m.pt) overriding the config default.
    """
    try:
        parsed = parse_information_path(information_path)
        if not parsed.product_root.exists():
            raise OnsiteRegistrationError(6, f"Product registration folder not found: {parsed.product_root}")
        yaml_path = resolve_registration_yaml(parsed)

        destination_root: Optional[Path] = None
        if output_folder:
            destination_root = Path(output_folder).expanduser() / parsed.stage / parsed.camera / parsed.printer_id

        output_root = build_onsite_training_data(
            registration_yaml_path=str(yaml_path),
            destination_root=str(destination_root) if destination_root else None,
            run_training=run_training,
            only_shots=[shot] if shot else None,
            only_models=[model] if model else None,
            model_variant=model_variant or None,
        )
        scope = (f"shot={shot!r}" if shot else "all shots") + (f", model={model!r}" if model else ", all models")
        scope += f", yolo={model_variant}" if model_variant else ""
        message = (
            f"onsite training completed ({scope})"
            if run_training
            else f"onsite build completed ({scope}, training skipped)"
        )
        return True, message, output_root, error_code_str(_ERROR_GROUP, 0)
    except OnsiteRegistrationError as exc:
        return False, str(exc), None, exc.code_str
    except Exception as exc:  # noqa: BLE001
        return False, str(exc), None, error_code_str(_ERROR_GROUP, 7)
