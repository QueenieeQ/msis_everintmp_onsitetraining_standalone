"""Studio train request: single model + YOLO version reach the trainer and are validated."""

from pathlib import Path
from unittest.mock import patch

import pytest

from onsite_training_studio.studio import products, train_options, train_runner

try:  # the API tests need a working fastapi/starlette pair
    from fastapi.testclient import TestClient

    from onsite_training_studio.studio import app as studio_app

    _API_OK = True
except Exception:  # noqa: BLE001 - e.g. fastapi/starlette version skew
    _API_OK = False

needs_api = pytest.mark.skipif(not _API_OK, reason="fastapi app cannot be created in this env")


@pytest.fixture()
def client():
    ref = products.ProductRef("mpfront", "realsense", "P1", root=Path("/tmp/P1"))
    with patch.object(studio_app, "_product", return_value=ref):
        yield TestClient(studio_app.create_app())


def _body(**kw):
    return {"stage": "mpfront", "camera": "realsense", "printer_id": "P1", **kw}


@needs_api
def test_options_listed_in_products_endpoint(client):
    data = client.get("/api/products").json()
    assert [m["value"] for m in data["models"]] == ["", "screw_hole", "patches", "screw_head"]
    assert "yolo11m.pt" in data["yolo_variants"]


@needs_api
def test_train_passes_model_and_variant(client):
    with patch.object(studio_app.train_jobs, "start_train", return_value="job1") as start:
        r = client.post("/api/train", json=_body(shot="left", model="screw_head", model_variant="yolo11m.pt"))
    assert r.status_code == 200
    kw = start.call_args.kwargs
    assert (kw["shot"], kw["model"], kw["model_variant"]) == ("left", "screw_head", "yolo11m.pt")


@needs_api
@pytest.mark.parametrize("extra", [{"model": "bogus"}, {"model_variant": "../x.pt"}])
def test_train_rejects_unknown_choices(client, extra):
    with patch.object(studio_app.train_jobs, "start_train") as start:
        r = client.post("/api/train", json=_body(**extra))
    assert r.status_code == 400
    start.assert_not_called()


def test_runner_forwards_selection_to_builder(tmp_path):
    with patch.object(train_runner, "parse_information_path") as parse, patch.object(
        train_runner, "resolve_registration_yaml", return_value=tmp_path / "r.yaml"
    ), patch.object(train_runner, "build_onsite_training_data", return_value="out") as build:
        parse.return_value.product_root = tmp_path
        ok, *_ = train_runner.run_training("p", True, None, "left", "screw_head", "yolo11l.pt")
    assert ok
    kw = build.call_args.kwargs
    assert kw["only_shots"] == ["left"] and kw["only_models"] == ["screw_head"]
    assert kw["model_variant"] == "yolo11l.pt"


def test_choices_cover_every_trainable_model_kind():
    from onsite_training_studio.training.model_selection import MODEL_KINDS

    assert set(train_options.model_values()) <= set(MODEL_KINDS)
    assert {"screw_hole", "patches", "screw_head"} == set(train_options.model_values())
