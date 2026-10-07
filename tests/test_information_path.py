"""Tests for onsite training entry via information_path."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from onsite_training_studio.training.information_path import (
    OnsiteRegistrationError,
    parse_information_path,
    process,
    resolve_registration_yaml,
)


def _product_tree(tmp_path: Path, *, stage="mpback", camera="realsense", printer="XM7-40"):
    root = tmp_path / "Product_Registration_Information" / stage / camera / printer
    (root / "configs").mkdir(parents=True)
    (root / "images").mkdir(parents=True)
    return root


def test_parse_absolute_information_path(tmp_path):
    product = _product_tree(tmp_path)
    parsed = parse_information_path(product)
    assert parsed.stage == "mpback"
    assert parsed.camera == "realsense"
    assert parsed.printer_id == "XM7-40"
    assert parsed.surface == "back"
    assert parsed.product_root == product.resolve()


def test_parse_mpfront_maps_surface_front(tmp_path):
    product = _product_tree(tmp_path, stage="mpfront")
    parsed = parse_information_path(product)
    assert parsed.surface == "front"
    assert parsed.stage == "mpfront"


def test_parse_rejects_missing_marker(tmp_path):
    bad = tmp_path / "other" / "realsense" / "XM7-40"
    bad.mkdir(parents=True)
    with pytest.raises(OnsiteRegistrationError, match="Product_Registration_Information"):
        parse_information_path(bad)


def test_parse_rejects_short_path(tmp_path):
    short = tmp_path / "Product_Registration_Information" / "mpback"
    short.mkdir(parents=True)
    with pytest.raises(OnsiteRegistrationError, match="stage/camera/printer"):
        parse_information_path(short)


def test_parse_strips_trailing_configs_segment(tmp_path):
    product = _product_tree(tmp_path)
    nested = product / "configs" / "registration_info.yaml"
    nested.write_text("printer_name: XM7-40\n")
    parsed = parse_information_path(nested)
    assert parsed.product_root == product.resolve()
    assert parsed.printer_id == "XM7-40"


def test_resolve_prefers_named_yaml(tmp_path):
    product = _product_tree(tmp_path)
    named = product / "configs" / "XM7-40_back_registration_info.yaml"
    legacy = product / "configs" / "registration_info.yaml"
    named.write_text("x: 1\n")
    legacy.write_text("x: 2\n")
    parsed = parse_information_path(product)
    assert resolve_registration_yaml(parsed) == named.resolve()


def test_resolve_falls_back_to_legacy(tmp_path):
    product = _product_tree(tmp_path)
    legacy = product / "configs" / "registration_info.yaml"
    legacy.write_text("x: 1\n")
    parsed = parse_information_path(product)
    assert resolve_registration_yaml(parsed) == legacy.resolve()


def test_resolve_missing_yaml_raises(tmp_path):
    product = _product_tree(tmp_path)
    parsed = parse_information_path(product)
    with pytest.raises(OnsiteRegistrationError, match="registration YAML|No registration"):
        resolve_registration_yaml(parsed)


def test_process_success_mocks_build(tmp_path):
    product = _product_tree(tmp_path)
    yaml_path = product / "configs" / "registration_info.yaml"
    yaml_path.write_text("printer_name: XM7-40\n")

    with patch(
        "onsite_training_studio.training.model.build_onsite_training_data",
        return_value="/out/root",
    ) as build:
        result = process(
            "registration_vision_training",
            str(product),
        )

    assert result["success"] is True
    assert result["output_root"] == "/out/root"
    assert result["message"] == "onsite training completed"
    assert result.get("error_code") == "115-000"
    build.assert_called_once()
    kwargs = build.call_args.kwargs
    assert Path(kwargs["registration_yaml_path"]) == yaml_path.resolve()
    assert kwargs["run_training"] is True


def test_process_success_mocks_build_no_yolo(tmp_path):
    product = _product_tree(tmp_path)
    yaml_path = product / "configs" / "registration_info.yaml"
    yaml_path.write_text("printer_name: XM7-40\n")

    with patch(
        "onsite_training_studio.training.model.build_onsite_training_data",
        return_value="/out/root",
    ) as build:
        result = process(
            "registration_vision_training",
            str(product),
            run_training=False,
        )

    assert result["success"] is True
    assert "skipped" in result["message"]
    assert build.call_args.kwargs["run_training"] is False


def test_process_bad_command():
    result = process("wrong_command", "/any/path")
    assert result["success"] is False
    assert result["output_root"] is None
    assert "Unsupported command" in result["message"]


def test_process_missing_path_returns_error(tmp_path):
    missing = tmp_path / "Product_Registration_Information" / "mpback" / "realsense" / "NOPE"
    result = process("registration_vision_training", str(missing))
    assert result["success"] is False
    assert result["output_root"] is None
    assert result["message"]


def test_process_empty_information_path():
    result = process("registration_vision_training", "")
    assert result["success"] is False
    assert "information_path" in result["message"].lower() or "empty" in result["message"].lower()


def test_model_reexports_information_path_process(tmp_path):
    """model.process is the framework import path and takes information_path."""
    from onsite_training_studio.training import model as onsite_model

    product = _product_tree(tmp_path)
    (product / "configs" / "registration_info.yaml").write_text("printer_name: XM7-40\n")

    with patch.object(onsite_model, "build_onsite_training_data", return_value="/out"):
        result = onsite_model.process(
            "registration_vision_training",
            str(product),
        )
    assert result["success"] is True
    assert result["output_root"] == "/out"


def test_parse_relative_path_is_taken_from_workspace(tmp_path):
    from onsite_training_studio import paths

    product = paths.registration_root() / "mpback" / "realsense" / "XM7-40"
    (product / "configs").mkdir(parents=True)
    parsed = parse_information_path("Product_Registration_Information/mpback/realsense/XM7-40")
    assert parsed.printer_id == "XM7-40"
    assert parsed.product_root == product.resolve()
