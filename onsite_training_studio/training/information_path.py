"""Parse Product_Registration_Information paths and run onsite training.

MQTT / framework entry uses ``information_path`` only, e.g.::

    Common/Product_Registration_Information/mpback/realsense/XM7-40
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from loguru import logger

from onsite_training_studio.errors import error_code_str, error_comment
from onsite_training_studio.paths import resolve_workspace_path
from onsite_training_studio.colors import green, red

PathLike = Union[str, Path]

_MARKER = "Product_Registration_Information"
_STAGE_TO_SURFACE = {"mpfront": "front", "mpback": "back"}
_VALID_CAMERAS = {"realsense", "omron", "Top_Camera"}
_COMMAND = "registration_vision_training"
_ERROR_GROUP = "onsite_registration"


class OnsiteRegistrationError(Exception):
    """Failure during onsite registration; suffix maps to onsite_registration 115-0XX codes."""

    def __init__(self, suffix: int, detail: str = "") -> None:
        self.suffix = int(suffix)
        self.detail = str(detail or "").strip()
        super().__init__(error_comment(_ERROR_GROUP, self.suffix, self.detail))

    @property
    def code_str(self) -> str:
        return error_code_str(_ERROR_GROUP, self.suffix)


@dataclass(frozen=True)
class ParsedInformationPath:
    product_root: Path
    stage: str
    camera: str
    printer_id: str
    surface: str


def parse_information_path(information_path: PathLike) -> ParsedInformationPath:
    """Extract stage/camera/printer_id/surface from an information_path."""
    text = str(information_path or "").strip()
    if not text:
        raise OnsiteRegistrationError(6, "information_path must be a non-empty path")

    resolved = resolve_workspace_path(text)
    parts = resolved.parts
    try:
        idx = parts.index(_MARKER)
    except ValueError as exc:
        raise OnsiteRegistrationError(
            6, f"Path must contain '{_MARKER}': {resolved}"
        ) from exc

    tail = parts[idx + 1 :]
    if len(tail) < 3:
        raise OnsiteRegistrationError(
            6, f"information_path needs stage/camera/printer after {_MARKER}, got: {resolved}"
        )

    stage, camera, printer_id = tail[0], tail[1], tail[2]
    if stage not in _STAGE_TO_SURFACE:
        raise OnsiteRegistrationError(6, f"Unknown stage '{stage}' (expected mpfront|mpback)")
    if camera not in _VALID_CAMERAS:
        raise OnsiteRegistrationError(6, f"Unknown camera '{camera}' (expected {_VALID_CAMERAS})")
    if not printer_id:
        raise OnsiteRegistrationError(6, "printer_id (product folder) is empty")

    product_root = Path(*parts[: idx + 4])
    return ParsedInformationPath(
        product_root=product_root,
        stage=stage,
        camera=camera,
        printer_id=printer_id,
        surface=_STAGE_TO_SURFACE[stage],
    )


def resolve_registration_yaml(parsed: ParsedInformationPath) -> Path:
    """Locate configs YAML under the product root."""
    configs_dir = parsed.product_root / "configs"
    candidates = [
        configs_dir / f"{parsed.printer_id}_{parsed.surface}_registration_info.yaml",
        configs_dir / "registration_info.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise OnsiteRegistrationError(
        1,
        f"No registration YAML found under {configs_dir} "
        f"(looked for: {[c.name for c in candidates]})",
    )


def infer(information_path: PathLike, run_training: bool = True) -> str:
    """Parse information_path, find registration YAML, run onsite build/train."""
    parsed = parse_information_path(information_path)
    if not parsed.product_root.exists():
        raise OnsiteRegistrationError(
            6, f"Product registration folder not found: {parsed.product_root}"
        )
    yaml_path = resolve_registration_yaml(parsed)
    logger.info(
        green(
            f"Resolved registration YAML for information_path={information_path!r}: "
            f"{yaml_path}"
        )
    )
    # Lazy import avoids circular import with model.py re-exports.
    from onsite_training_studio.training.model import build_onsite_training_data

    return build_onsite_training_data(
        registration_yaml_path=str(yaml_path),
        run_training=run_training,
    )


def process(
    command: str,
    information_path: PathLike,
    *,
    run_training: bool = True,
) -> dict:
    """Framework entry: never raises; returns success/output_root/message/error_code."""
    try:
        if command != _COMMAND:
            raise OnsiteRegistrationError(6, f"Unsupported command: {command!r}")
        if not str(information_path or "").strip():
            raise OnsiteRegistrationError(6, "information_path must be a non-empty path")

        output_root = infer(
            information_path=information_path,
            run_training=run_training,
        )
        return {
            "success": True,
            "output_root": output_root,
            "message": "onsite training completed"
            if run_training
            else "onsite build completed (training skipped)",
            "error_code": error_code_str(_ERROR_GROUP, 0),
        }
    except OnsiteRegistrationError as exc:
        logger.error(
            red(
                f"process failed for command={command!r} "
                f"information_path={information_path!r}: {exc}"
            )
        )
        return {
            "success": False,
            "output_root": None,
            "message": str(exc),
            "error_code": exc.code_str,
        }
    except Exception as exc:
        logger.error(
            red(
                f"process failed for command={command!r} "
                f"information_path={information_path!r}: {exc}"
            )
        )
        return {
            "success": False,
            "output_root": None,
            "message": str(exc),
            "error_code": error_code_str(_ERROR_GROUP, 7),
        }
