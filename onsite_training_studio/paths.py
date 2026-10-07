"""Workspace layout — the only place that knows where files live.

Everything the tool reads or writes sits under ONE workspace folder, so the
whole project can be copied to another PC and pointed at its own data:

    <workspace>/
        Product_Registration_Information/<stage>/<camera>/<product>/
            configs/<product>_<front|back>_registration_info.yaml
            images/<shot>.png                      (input: images + boxes)
        output/<stage>/<camera>/<product>/<shot>/   (output: weights, labels, ...)
        logs/onsite_training/                       (training logs)
        runs/  tmp/                                 (ultralytics runs / scratch datasets)
        onsite_training.yaml                        (training parameters)

Resolution order for the workspace: ``set_workspace()`` (CLI ``--workspace``)
> ``ONSITE_WORKSPACE`` env var > ``<project>/workspace``.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "ONSITE_WORKSPACE"
REGISTRATION_DIRNAME = "Product_Registration_Information"
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_workspace(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    os.environ[ENV_VAR] = str(resolved)
    return resolved


def workspace_root() -> Path:
    env = os.environ.get(ENV_VAR, "").strip()
    root = Path(env).expanduser() if env else _PROJECT_ROOT / "workspace"
    return root.resolve()


def registration_root() -> Path:
    return workspace_root() / REGISTRATION_DIRNAME


def output_root() -> Path:
    return workspace_root() / "output"


def logs_dir() -> Path:
    return workspace_root() / "logs"


def tmp_dir() -> Path:
    """Scratch space for generated datasets (deleted after each training run)."""
    return workspace_root() / "tmp"


def config_path() -> Path:
    return workspace_root() / "onsite_training.yaml"


def resolve_workspace_path(path_value, *, root: Path | None = None) -> Path:
    """Absolute paths pass through; relative ones are taken from the workspace."""
    path_value = Path(path_value).expanduser()
    if path_value.is_absolute():
        return path_value.resolve()
    return ((root if root is not None else workspace_root()) / path_value).resolve()
