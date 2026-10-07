"""Screw-head (label S) support in onsite training: S is parsed, written to its
own labels file, and trained as a separate model whose weights live under
weights/screw_head/screw_head_detection.pt — H keeps its own weight.pt flow.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

cv2 = pytest.importorskip("cv2")

from onsite_training_studio.training.model import ReferenceDataManager
from onsite_training_studio.training.screw_head import (
    SCREW_HEAD_LABELS_FILE,
    SCREW_HEAD_WEIGHTS_FILENAME,
    SCREW_HEAD_WEIGHTS_SUBDIR,
)

_BUILDER = "onsite_training_studio.training.mixins.reference_tree_builder"
_LOGS = "onsite_training_studio.paths.logs_dir"


def _registration(tmp_path: Path, shot_block: dict) -> Path:
    root = tmp_path / "Product_Registration_Information" / "mpfront" / "realsense" / "XM7-S"
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / "images" / "left.png"), np.full((60, 60, 3), 200, dtype=np.uint8))
    yaml_path = root / "configs" / "XM7-S_front_registration_info.yaml"
    yaml_path.write_text(
        yaml.safe_dump({"left": {"image_path": "images/left.png", **shot_block}}), encoding="utf-8"
    )
    return yaml_path


def _build(tmp_path, shot_block, run_training, **patches):
    yaml_path = _registration(tmp_path, shot_block)
    out = tmp_path / "out"
    mgr = ReferenceDataManager(reference_folder=str(out), create_root_dirs=False)
    mgr.build_reference_tree_from_registration_yaml_file(
        yaml_file_path=str(yaml_path), destination_root=str(out), run_training=run_training
    )
    return out / "left"


def _trainer_patch(*, roi=None, hole=None, patches=None, head=None):
    return patch(
        f"{_BUILDER}._load_trainers",
        return_value={
            "roi": roi or MagicMock(),
            "screw_hole": hole or MagicMock(),
            "patches": patches or MagicMock(),
            "screw_head": head or MagicMock(),
        },
    )


def test_s_label_is_written_separately_from_h(tmp_path):
    shot = _build(
        tmp_path,
        {"H": {"1": {"bbox": [30, 30, 40, 40]}}, "S": {"1": {"bbox": [5, 5, 15, 15]}}},
        run_training=False,
    )
    assert (shot / "labels" / SCREW_HEAD_LABELS_FILE).read_text().split() == ["0", "5", "5", "15", "15"]
    assert (shot / "labels" / "ref_user.txt").read_text().split() == ["0", "30", "30", "40", "40"]

    cfg = json.loads((shot / "config" / "ref_user_extracted.json").read_text())
    assert sorted(a["label"] for a in cfg["annotations"]) == ["H", "S"]
    assert cfg["screw_head_weight_path"].endswith(
        f"weights/{SCREW_HEAD_WEIGHTS_SUBDIR}/{SCREW_HEAD_WEIGHTS_FILENAME}"
    )
    assert cfg["weight_path"].endswith("weights/weight.pt")  # H weights path unchanged

    mask = cv2.imread(str(shot / "patches" / "mask_ref" / "mask_ref_user.png"), cv2.IMREAD_GRAYSCALE)
    assert mask[35, 35] == 255 and mask[10, 10] == 0  # S is not part of the hole mask
    assert list((shot / "patches" / "screw_head").glob("ref_user_S_0_*.png"))


def test_shot_without_s_has_no_screw_head_artifacts(tmp_path):
    shot = _build(tmp_path, {"H": {"1": {"bbox": [30, 30, 40, 40]}}}, run_training=False)
    assert not (shot / "labels" / SCREW_HEAD_LABELS_FILE).exists()
    cfg = json.loads((shot / "config" / "ref_user_extracted.json").read_text())
    assert "screw_head_weight_path" not in cfg


def test_s_only_shot_is_accepted_and_trains_only_screw_head(tmp_path):
    head, hole = MagicMock(), MagicMock()
    with _trainer_patch(hole=hole, head=head):
        _build(tmp_path, {"S": {"1": {"bbox": [5, 5, 15, 15]}}}, run_training=True)
    head.assert_called_once()
    assert head.call_args.kwargs["printer_name"] == "left"
    hole.assert_not_called()


def test_h_and_s_train_separate_models(tmp_path):
    head, hole = MagicMock(), MagicMock()
    with _trainer_patch(hole=hole, head=head):
        _build(
            tmp_path,
            {"H": {"1": {"bbox": [30, 30, 40, 40]}}, "S": {"1": {"bbox": [5, 5, 15, 15]}}},
            run_training=True,
        )
    head.assert_called_once()
    hole.assert_called_once()


def test_missing_screw_head_trainer_fails_the_shot(tmp_path):
    from onsite_training_studio.training.information_path import OnsiteRegistrationError

    with patch(f"{_BUILDER}._load_trainers", side_effect=ImportError("no torch")):
        with pytest.raises(OnsiteRegistrationError):
            _build(tmp_path, {"S": {"1": {"bbox": [5, 5, 15, 15]}}}, run_training=True)


def test_screw_head_trainer_saves_to_dedicated_weights_name(tmp_path):
    from onsite_training_studio.training import training_screw_head as mod

    captured = {}

    def fake_train(**kw):
        captured.update(kw)
        return tmp_path / "w.pt"

    shot = tmp_path / "left"
    (shot / "images").mkdir(parents=True)
    (shot / "labels").mkdir()
    (shot / "labels" / SCREW_HEAD_LABELS_FILE).write_text("0 1 1 5 5\n")
    cfg = {"dataset": {}, "train": {}, "augment": {}}
    with patch.object(mod, "load_yaml_config", return_value=cfg), patch.object(
        mod, "onsite_yaml_path", return_value="x.yaml"
    ), patch.object(mod, "generate_dataset", return_value=tmp_path / "ds") as gen, patch.object(
        mod, "train_yolo11s", side_effect=fake_train
    ), patch.object(mod, "setup_logger", return_value=MagicMock()), patch(
        _LOGS, return_value=tmp_path
    ):
        mod.train_screw_head_one_shot("left", str(tmp_path))

    assert gen.call_args.kwargs["class_name"] == "screw_head"
    assert captured["weights_subdir"] == Path(SCREW_HEAD_WEIGHTS_SUBDIR)
    assert captured["weights_filename"] == SCREW_HEAD_WEIGHTS_FILENAME
    assert captured["run_prefix"] == "screw_head"


# ---- single-model selection + YOLO version -------------------------------------------------

_BOTH = {"H": {"1": {"bbox": [30, 30, 40, 40]}}, "S": {"1": {"bbox": [5, 5, 15, 15]}}}


def _trainers(**over):
    t = {k: MagicMock() for k in ("roi", "hole", "patch", "head")}
    t.update(over)
    return t


def _build_with(tmp_path, block, trainers, **kw):
    out = tmp_path / "out"
    yaml_path = _registration(tmp_path, block)
    mgr = ReferenceDataManager(reference_folder=str(out), create_root_dirs=False)
    with _trainer_patch(
        roi=trainers["roi"], hole=trainers["hole"], patches=trainers["patch"], head=trainers["head"]
    ):
        mgr.build_reference_tree_from_registration_yaml_file(
            yaml_file_path=str(yaml_path), destination_root=str(out), run_training=True, **kw
        )
    return out / "left"


@pytest.mark.parametrize(
    "model,expected",
    [("screw_hole", "hole"), ("patches", "patch"), ("screw_head", "head")],
)
def test_single_model_trains_only_that_model(tmp_path, model, expected):
    t = _trainers()
    _build_with(tmp_path, _BOTH, t, only_models=[model])
    for name, trainer in t.items():
        assert trainer.called == (name == expected), name


def test_single_model_keeps_existing_weights(tmp_path):
    t = _trainers()
    shot = _build_with(tmp_path, _BOTH, t)  # first full build
    keep = shot / "weights" / "weight.pt"
    keep.write_bytes(b"hole-weights")
    _build_with(tmp_path, _BOTH, _trainers(), only_models=["screw_head"])
    assert keep.read_bytes() == b"hole-weights"  # not archived away


def test_selected_model_without_annotations_fails(tmp_path):
    from onsite_training_studio.training.information_path import OnsiteRegistrationError

    with pytest.raises(OnsiteRegistrationError):
        _build_with(tmp_path, {"H": {"1": {"bbox": [30, 30, 40, 40]}}}, _trainers(), only_models=["screw_head"])


def test_model_variant_is_passed_to_every_selected_trainer(tmp_path):
    t = _trainers()
    _build_with(tmp_path, _BOTH, t, model_variant="yolo11m.pt")
    for name in ("hole", "patch", "head"):
        assert t[name].call_args.kwargs["model_variant"] == "yolo11m.pt"


def test_default_variant_adds_no_kwarg(tmp_path):
    t = _trainers()
    _build_with(tmp_path, _BOTH, t)
    assert "model_variant" not in t["head"].call_args.kwargs


@pytest.mark.parametrize("bad", ["../evil.pt", "yolo11s", "a b.pt"])
def test_invalid_variant_rejected(bad):
    from onsite_training_studio.training.information_path import OnsiteRegistrationError
    from onsite_training_studio.training.model_selection import normalize_model_variant

    with pytest.raises(OnsiteRegistrationError):
        normalize_model_variant(bad)


def test_unknown_model_rejected():
    from onsite_training_studio.training.information_path import OnsiteRegistrationError
    from onsite_training_studio.training.model_selection import normalize_only_models

    with pytest.raises(OnsiteRegistrationError):
        normalize_only_models(["screw_hole", "bogus"])


def test_trainer_applies_model_variant_to_train_cfg(tmp_path):
    from onsite_training_studio.training import training_screw_head as mod

    seen = {}
    shot = tmp_path / "left"
    (shot / "images").mkdir(parents=True)
    (shot / "labels").mkdir()
    (shot / "labels" / SCREW_HEAD_LABELS_FILE).write_text("0 1 1 5 5\n")
    cfg = {"dataset": {}, "train": {"model_variant": "yolo11s.pt"}, "augment": {}}
    with patch.object(mod, "load_yaml_config", return_value=cfg), patch.object(
        mod, "onsite_yaml_path", return_value="x.yaml"
    ), patch.object(mod, "generate_dataset", return_value=tmp_path / "ds"), patch.object(
        mod, "train_yolo11s", side_effect=lambda **kw: seen.update(kw) or tmp_path / "w.pt"
    ), patch.object(mod, "setup_logger", return_value=MagicMock()), patch(
        _LOGS, return_value=tmp_path
    ):
        mod.train_screw_head_one_shot("left", str(tmp_path), model_variant="yolo11l.pt")
    assert seen["train_cfg"]["model_variant"] == "yolo11l.pt"


def test_patch_trainer_fails_when_every_hole_fails(tmp_path):
    from onsite_training_studio.training import training_patch as mod

    shot = tmp_path / "left"
    (shot / "images").mkdir(parents=True)
    (shot / "labels").mkdir()
    cv2.imwrite(str(shot / "images" / "ref_user.png"), np.full((60, 60, 3), 200, dtype=np.uint8))
    (shot / "labels" / "ref_user.txt").write_text("0 10 10 30 30\n")
    cfg = {"dataset": {}, "train": {}, "augment": {}}
    with patch.object(mod, "load_yaml_config", return_value=cfg), patch.object(
        mod, "generate_patch_dataset", side_effect=RuntimeError("boom")
    ), patch.object(mod, "setup_logger", return_value=MagicMock()):
        with pytest.raises(RuntimeError, match="every hole"):
            mod.train_patch_one_shot("left", str(tmp_path))
