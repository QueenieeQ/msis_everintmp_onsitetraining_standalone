"""Load/save bbox annotations inside a registration_info.yaml shot block.

YAML shape (see Common/Product_Registration_Information/**/configs/*.yaml)::

    left:
      image_path: images\\left.png
      H:
        '1':
          bbox: [x1, y1, x2, y2]
        '2':
          bbox: [x1, y1, x2, y2]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

SHOT_NAMES = ("left", "right", "closer")
# H = screw hole (trains weights/weight.pt for model_1);
# S = screw head (trained as its own model: weights/screw_head/screw_head_detection.pt).
LABELS = ("H", "S", "R", "X", "Y")


def load_yaml(yaml_path: Path) -> Dict[str, Any]:
    text = Path(yaml_path).read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


def save_yaml(yaml_path: Path, data: Dict[str, Any]) -> None:
    text = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    Path(yaml_path).write_text(text, encoding="utf-8")


def ensure_shot(data: Dict[str, Any], shot_name: str) -> Dict[str, Any]:
    """Return data[shot_name], creating a bare {image_path: ...} block if missing."""
    shot_data = data.get(shot_name)
    if not isinstance(shot_data, dict):
        shot_data = {"image_path": f"images/{shot_name}.png"}
        data[shot_name] = shot_data
    return shot_data


def shot_image_path(product_root: Path, shot_data: Dict[str, Any], shot_name: str) -> Path:
    """Resolve product_root/images/<...> for this shot's image_path."""
    raw = shot_data.get("image_path") or f"images/{shot_name}.png"
    raw = str(raw).replace("\\", "/")
    parts = Path(raw).parts
    if parts and parts[0].lower() == "images":
        return product_root / Path(*parts)
    return product_root / "images" / Path(raw).name


def _sort_key(inst_id: str):
    return (0, int(inst_id)) if inst_id.isdigit() else (1, inst_id)


def get_boxes(shot_data: Dict[str, Any], label: str) -> List[Tuple[str, List[float]]]:
    group = shot_data.get(label)
    if not isinstance(group, dict):
        return []
    boxes = [
        (str(inst_id), [float(v) for v in inst["bbox"]])
        for inst_id, inst in group.items()
        if isinstance(inst, dict) and isinstance(inst.get("bbox"), list)
    ]
    boxes.sort(key=lambda item: _sort_key(item[0]))
    return boxes


def set_boxes(shot_data: Dict[str, Any], label: str, boxes: List[List[float]]) -> None:
    """Replace all instances of `label` in-place, renumbered '1', '2', ...."""
    if boxes:
        shot_data[label] = {
            str(i + 1): {"bbox": [int(round(v)) for v in box]}
            for i, box in enumerate(boxes)
        }
    else:
        shot_data.pop(label, None)
