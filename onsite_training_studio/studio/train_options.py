"""Selectable training options exposed by the studio UI."""

from __future__ import annotations

# (value sent to the API, label shown in the UI). "" = every model the shot has data for.
MODEL_CHOICES = (
    ("", "All models (everything the shot has boxes for)"),
    ("screw_hole", "Screw hole (H) -> weights/weight.pt"),
    ("patches", "Patches (H) -> weights/patches_weights/"),
    ("screw_head", "Screw head (S) -> weights/screw_head/"),
)

# Ultralytics checkpoints to fine-tune from. "" = use the onsite YAML default.
YOLO_VARIANTS = (
    "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt",
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
)


def model_values() -> tuple:
    return tuple(v for v, _ in MODEL_CHOICES if v)
