"""Choose which per-shot models onsite training builds, and with which YOLO.

Kinds (per shot) and the annotation each one trains from:

    screw_hole  H  -> weights/weight.pt                  (the main detector weights)
    patches     H  -> weights/patches_weights/hole_<i>/
    screw_head  S  -> weights/screw_head/screw_head_detection.pt
    roi         P  -> weights/weight.pt (ROI shots)

``only_models=None`` keeps the historical behavior (train everything the
shot has annotations for). A non-empty selection trains just those kinds and
fails loudly when a selected kind has nothing to train from.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, Iterable, Optional

from onsite_training_studio.training.information_path import OnsiteRegistrationError

MODEL_KINDS = ("screw_hole", "patches", "screw_head", "roi")
_VARIANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*\.pt$")


def normalize_only_models(only_models: Optional[Iterable[str]]) -> Optional[frozenset]:
    if not only_models:
        return None
    wanted = {str(m).strip().lower() for m in only_models}
    unknown = wanted - set(MODEL_KINDS)
    if unknown:
        raise OnsiteRegistrationError(6, f"Unknown model(s) {sorted(unknown)}; expected {list(MODEL_KINDS)}")
    return frozenset(wanted)


def normalize_model_variant(model_variant: Optional[str]) -> Optional[str]:
    """Plain ultralytics weight name (e.g. ``yolo11m.pt``) or None for config default."""
    text = str(model_variant or "").strip()
    if not text:
        return None
    if not _VARIANT_RE.match(text):
        raise OnsiteRegistrationError(6, f"Invalid YOLO model_variant: {text!r}")
    return text


def train_selected_models(
    *,
    only_models: Optional[frozenset],
    model_variant: Optional[str],
    has_data: Dict[str, bool],
    trainers: Dict[str, Optional[Callable]],
    calls: Dict[str, Callable[[Callable, dict], object]],
    on_start: Callable[[str], None],
) -> None:
    """Run each wanted trainer; ``calls[kind](trainer, extra_kwargs)`` invokes it.

    Raises RuntimeError when an explicitly selected kind has no annotations or
    no trainer; unselected kinds are skipped silently.
    """
    extra = {"model_variant": model_variant} if model_variant else {}
    for kind in MODEL_KINDS:
        if only_models is not None and kind not in only_models:
            continue
        if not has_data.get(kind):
            if only_models is not None:
                raise RuntimeError(f"{kind}: shot has no annotations to train from")
            continue
        trainer = trainers.get(kind)
        if trainer is None:
            raise RuntimeError(f"{kind} trainer unavailable (is torch/ultralytics installed?)")
        on_start(kind)
        calls[kind](trainer, extra)
