"""Tests for _ReferenceTreeBuilderMixin.build_reference_tree_from_registration_yaml_file."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import yaml

cv2 = pytest.importorskip("cv2")

from onsite_training_studio.training.model import ReferenceDataManager
from onsite_training_studio.training.mixins.reference_tree_builder import (
    _shots_config_from_registration,
)


def _write_registration_yaml_and_image(tmp_path: Path, printer_id: str) -> Path:
    """Build a minimal new-format registration tree:
    <tmp_path>/Product_Registration_Information/mpfront/realsense/<printer_id>/
        configs/<printer_id>_front_registration_info.yaml
        images/roi.png
    """
    product_root = tmp_path / "Product_Registration_Information" / "mpfront" / "realsense" / printer_id
    (product_root / "configs").mkdir(parents=True)
    (product_root / "images").mkdir(parents=True)

    image_path = product_root / "images" / "roi.png"
    cv2.imwrite(str(image_path), np.zeros((20, 20, 3), dtype=np.uint8))

    yaml_path = product_root / "configs" / f"{printer_id}_front_registration_info.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "roi": {
                    "image_path": "images/roi.png",
                    "P": {"bbox": [1, 1, 10, 10]},
                }
            }
        ),
        encoding="utf-8",
    )
    return yaml_path


def test_shots_config_from_registration_derives_holes_and_defaults_unregistered_shots():
    """Mirrors the R410 front example: roi -> 0 holes (always, roi_detection);
    closer/left -> hole count from the H dict; a screw shot with no H
    annotations yet (image only) is kept as screw_detection with
    expected_holes=0 rather than dropped.
    """
    parsed_shots = [
        ("roi", {"image_path": "images/roi.png", "P": {"1": {"bbox": [192, 137, 922, 509]}}}),
        (
            "closer",
            {
                "image_path": "images/closer.png",
                "H": {
                    "1": {"bbox": [363, 324, 389, 350]},
                    "2": {"bbox": [531, 323, 552, 344]},
                    "3": {"bbox": [610, 303, 633, 326]},
                },
            },
        ),
        (
            "left",
            {
                "image_path": "images/left.png",
                "H": {
                    "1": {"bbox": [360, 309, 386, 335]},
                    "2": {"bbox": [526, 327, 548, 349]},
                    "3": {"bbox": [607, 334, 631, 358]},
                },
            },
        ),
        ("right", {"image_path": "images/right.png"}),
    ]

    shots_config = _shots_config_from_registration(
        parsed_shots, printer_id="Dummy", camera_folder="realsense", side_folder="mpfront"
    )

    assert shots_config == {
        "realsense_front_roi": {
            "shot_type": "roi_detection",
            "expected_holes": 0,
            "ref_json_subpath": "mpfront/realsense/Dummy/roi/config/ref_user_extracted.json",
        },
        "realsense_front_closer": {
            "shot_type": "screw_detection",
            "expected_holes": 3,
            "ref_json_subpath": "mpfront/realsense/Dummy/closer/config/ref_user_extracted.json",
        },
        "realsense_front_left": {
            "shot_type": "screw_detection",
            "expected_holes": 3,
            "ref_json_subpath": "mpfront/realsense/Dummy/left/config/ref_user_extracted.json",
        },
        # "right" has no H/P yet (image only) — kept as screw_detection with
        # expected_holes=0 rather than dropped.
        "realsense_front_right": {
            "shot_type": "screw_detection",
            "expected_holes": 0,
            "ref_json_subpath": "mpfront/realsense/Dummy/right/config/ref_user_extracted.json",
        },
    }


def test_build_writes_shots_summary_and_defaults_to_workspace_output(tmp_path):
    from onsite_training_studio import paths

    yaml_path = _write_registration_yaml_and_image(paths.workspace_root(), "XM7-TEST")
    mgr = ReferenceDataManager(reference_folder=str(tmp_path / "scratch"), create_root_dirs=False)

    result = mgr.build_reference_tree_from_registration_yaml_file(
        yaml_file_path=str(yaml_path), run_training=False
    )

    expected = paths.output_root() / "mpfront" / "realsense" / "XM7-TEST"
    assert Path(result) == expected
    shots = yaml.safe_load((expected / "shots_config.yaml").read_text())["shots"]
    assert shots["realsense_front_roi"]["shot_type"] == "roi_detection"


def test_relative_paths_follow_model_1_convention(tmp_path):
    """ref_user_extracted.json paths stay drop-in for model_1's data folder."""
    import json

    from onsite_training_studio import paths

    yaml_path = _write_registration_yaml_and_image(paths.workspace_root(), "XM7-TEST")
    mgr = ReferenceDataManager(reference_folder=str(tmp_path / "scratch"), create_root_dirs=False)
    out = Path(mgr.build_reference_tree_from_registration_yaml_file(str(yaml_path), run_training=False))

    cfg = json.loads((out / "roi" / "config" / "ref_user_extracted.json").read_text())
    base = "data/models/model_1/mpfront/realsense/XM7-TEST/roi"
    assert cfg["image"] == f"{base}/images/ref_user.png"
    assert cfg["weight_path"] == f"{base}/weights/weight.pt"


def test_rebuild_archives_previous_output(tmp_path):
    from onsite_training_studio import paths

    yaml_path = _write_registration_yaml_and_image(paths.workspace_root(), "XM7-TEST")
    mgr = ReferenceDataManager(reference_folder=str(tmp_path / "scratch"), create_root_dirs=False)
    out = Path(mgr.build_reference_tree_from_registration_yaml_file(str(yaml_path), run_training=False))
    (out / "marker.txt").write_text("old")
    mgr.build_reference_tree_from_registration_yaml_file(str(yaml_path), run_training=False)

    assert not (out / "marker.txt").exists()
    assert list((out.parent / "old_registration_data").glob("XM7-TEST-*/marker.txt"))
