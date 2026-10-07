"""Discover and create Product_Registration_Information products."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from onsite_training_studio.paths import registration_root

from . import registration_io

_MARKER = "Product_Registration_Information"
_STAGE_TO_SURFACE = {"mpfront": "front", "mpback": "back"}
STAGES = tuple(_STAGE_TO_SURFACE)
CAMERAS = ("realsense", "omron", "Top_Camera")
_PRINTER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class InvalidProductError(ValueError):
    """Raised for a bad stage/camera/printer_id, or a product that already exists."""


@dataclass(frozen=True)
class ProductRef:
    stage: str
    camera: str
    printer_id: str
    root: Path

    @property
    def information_path(self) -> str:
        return str(self.root)

    def registration_yaml_target(self) -> Path:
        """Canonical registration YAML path, whether or not it exists yet."""
        surface = _STAGE_TO_SURFACE.get(self.stage, self.stage)
        return self.root / "configs" / f"{self.printer_id}_{surface}_registration_info.yaml"

    def registration_yaml(self) -> Path:
        surface = _STAGE_TO_SURFACE.get(self.stage, self.stage)
        configs_dir = self.root / "configs"
        for name in (
            f"{self.printer_id}_{surface}_registration_info.yaml",
            "registration_info.yaml",
        ):
            candidate = configs_dir / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No registration YAML found under {configs_dir}")


def discover_products() -> List[ProductRef]:
    base = registration_root()
    found: List[ProductRef] = []
    if not base.is_dir():
        return found
    for stage in STAGES:
        for camera in CAMERAS:
            camera_dir = base / stage / camera
            if not camera_dir.is_dir():
                continue
            for printer_dir in sorted(camera_dir.iterdir()):
                if printer_dir.is_dir() and (printer_dir / "configs").is_dir():
                    found.append(ProductRef(stage, camera, printer_dir.name, printer_dir))
    return found


def find_product(stage: str, camera: str, printer_id: str) -> ProductRef:
    root = registration_root() / stage / camera / printer_id
    if not (root / "configs").is_dir():
        raise FileNotFoundError(f"Product not found: {root}")
    return ProductRef(stage, camera, printer_id, root)


def create_product(stage: str, camera: str, printer_id: str) -> ProductRef:
    """Create a new product folder + a bare registration YAML (no images yet).

    Shots are seeded with only ``image_path`` (no boxes) so the studio can
    immediately offer left/right/closer for image upload and annotation.
    """
    if stage not in STAGES:
        raise InvalidProductError(f"Unknown stage {stage!r} (expected {STAGES})")
    if camera not in CAMERAS:
        raise InvalidProductError(f"Unknown camera {camera!r} (expected {CAMERAS})")
    if not _PRINTER_ID_RE.match(printer_id):
        raise InvalidProductError(
            f"Invalid product name {printer_id!r}: use letters/digits/-/_/. only"
        )

    root = registration_root() / stage / camera / printer_id
    if (root / "configs").is_dir():
        raise InvalidProductError(f"Product already exists: {root}")

    product = ProductRef(stage, camera, printer_id, root)
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(parents=True, exist_ok=True)

    data = {shot: {"image_path": f"images/{shot}.png"} for shot in registration_io.SHOT_NAMES}
    registration_io.save_yaml(product.registration_yaml_target(), data)
    return product
