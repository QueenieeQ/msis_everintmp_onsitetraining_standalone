# training_defaults.py
# ------------------------------------------------------------
# Shared onsite-training config: the ONSITE_YAML_PATH location, default
# dataset/train/augment values (DEFAULT_CFG; auto-created file content is
# _onsite_yaml_template.yaml), and the YAML load/merge/resolve helpers
# (ensure_yaml, deep_merge, resolve_param, load_yaml_config) used by
# training.py (ROI), training_screw.py (screw), and training_patch.py
# (patch). Split out of those files so each stays under 300 lines and
# there is a single source of truth for the config shape all three
# profiles ("roi"/"screw"/"patch") share.
#
# Structured parameters: any leaf may be either a plain
# scalar/list OR a dict carrying {name, access, definition, Min, Max,
# Default, Best, Step, Current} for GUI editing — resolve_param/_resolve_tree
# collapse the latter to its Current (falling back to Default) so callers
# (training.py/training_screw.py/training_patch.py) only ever see plain
# scalars, unchanged from before this was introduced.
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from onsite_training_studio.paths import config_path, resolve_workspace_path


def onsite_yaml_path() -> Path:
    """``<workspace>/onsite_training.yaml`` (auto-created from the template on first use)."""
    return config_path()


# ============================================================
# DEFAULT CONFIG (used as fallback; YAML overrides it)
# ============================================================
DEFAULT_CFG: Dict[str, Any] = {
    "dataset": {
        "n_images": 500,
        "val_ratio": 0.10,
        "imgsz": 640,
        "seed": 123,
    },
    "train": {
        "epochs": 60,
        "batch": 16,
        "device": "0",       # "0" or "cpu"
        "workers": 4,
        "patience": 12,
        "project_dir": "runs",
        "delete_runs_dir": False,
        # cuDNN 9.x's lazy dynamic-symbol loader crashes on this machine
        # (SIGABRT: "Invalid handle. Cannot load symbol cudnnGetVersion") —
        # disabling amp alone only delayed the crash to the first real conv
        # pass, so cuDNN itself is off by default; PyTorch falls back to its
        # native (non-cuDNN) CUDA conv kernels. Flip both back on once
        # verified safe on the target machine's cuDNN build.
        "amp": False,
        "cudnn_enabled": False,
    },
    "augment": {
        # probabilities per sample
        "p_brightness": 0.55,
        "p_translate": 0.55,
        "p_noise": 0.45,
        "p_blur": 0.35,
        "p_rotate": 0.45,
        "p_shear": 0.30,
        "p_scale": 0.55,

        # ranges
        "brightness": {"gain_min": 0.75, "gain_max": 1.25, "bias_min": -18, "bias_max": 18},
        "translate": {"frac_x": 0.10, "frac_y": 0.10},
        "rotate": {"deg_min": -12, "deg_max": 12},
        "scale": {"min": 0.85, "max": 1.20},
        "shear": {"shx_min": -0.08, "shx_max": 0.08, "shy_min": -0.05, "shy_max": 0.05},
        "noise": {"sigma_min": 3.0, "sigma_max": 18.0},
        "blur": {"kernels": [3, 5, 7]},
    },
}


# ============================================================
# YAML TEMPLATE (auto-created if missing)
# ------------------------------------------------------------
# The template text lives in _onsite_yaml_template.yaml (declarative
# resource, not application source) rather than a Python string, so this
# file stays under the 200-line limit. It carries roi:/screw:/patch:
# sections in the model_1.yaml UI-editable shape (each leaf param is a
# dict with name/access/definition/Min/Max/Default/Best/Step/Current) so
# the GUI's generic OTF parameter editor can read/modify it like model_1.
# ============================================================
_TEMPLATE_YAML_PATH = Path(__file__).with_name("_onsite_yaml_template.yaml")


def ensure_yaml(yaml_path: Path) -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    if not yaml_path.exists():
        yaml_path.write_text(_TEMPLATE_YAML_PATH.read_text(encoding="utf-8"), encoding="utf-8")


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override onto base (override wins)."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_param(value: Any) -> Any:
    """Collapse a model_1-style structured param dict to its runtime value.

    Mirrors ParamLoader.get() (io_param_loader.py): a dict carrying 'Current'
    returns that; a dict carrying only 'Default' falls back to it; anything
    else (plain scalar, list, or a container dict with neither key) passes
    through unchanged.
    """
    if isinstance(value, dict):
        if "Current" in value:
            return value["Current"]
        if "Default" in value:
            return value["Default"]
    return value


def _resolve_tree(node: Any) -> Any:
    """Recursively apply resolve_param to every structured-param leaf.

    A dict is a param leaf (collapses to a scalar/list) if it directly
    carries 'Current' or 'Default'; otherwise it's a container (dataset/
    train/augment/brightness/...) and each of its values is resolved in turn.
    """
    if isinstance(node, dict):
        if "Current" in node or "Default" in node:
            return resolve_param(node)
        return {k: _resolve_tree(v) for k, v in node.items()}
    return node


def load_yaml_config(yaml_path: Path, profile: str) -> Dict[str, Any]:
    cfg = _load_yaml_config(yaml_path, profile)
    # Ultralytics run folders live in the workspace, never relative to the cwd.
    train = cfg.setdefault("train", {})
    train["project_dir"] = str(resolve_workspace_path(train.get("project_dir", "runs")))
    return cfg


def _load_yaml_config(yaml_path: Path, profile: str) -> Dict[str, Any]:
    ensure_yaml(yaml_path)

    with yaml_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # Start from default values
    cfg = dict(DEFAULT_CFG)

    # New style:
    # parameters.yaml has roi:/screw:/patch:
    if profile in raw and isinstance(raw[profile], dict):
        cfg = deep_merge(cfg, raw[profile])
        return _resolve_tree(cfg)

    # Backward compatibility:
    # old style parameters.yaml has dataset/train/augment directly at root
    root_override = {
        k: raw[k]
        for k in ("dataset", "train", "augment")
        if k in raw and isinstance(raw[k], dict)
    }

    return _resolve_tree(deep_merge(cfg, root_override))
