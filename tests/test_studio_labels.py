"""Registration studio exposes the S (screw head) label next to H (screw hole)."""

from onsite_training_studio.studio import registration_io as rio


def test_s_label_is_offered_alongside_h():
    assert "H" in rio.LABELS and "S" in rio.LABELS


def test_s_boxes_round_trip_without_touching_h(tmp_path):
    data = {"left": {"image_path": "images/left.png", "H": {"1": {"bbox": [30, 30, 40, 40]}}}}
    shot = rio.ensure_shot(data, "left")
    rio.set_boxes(shot, "S", [[5, 5, 15, 15], [20, 5, 30, 15]])

    path = tmp_path / "reg.yaml"
    rio.save_yaml(path, data)
    shot = rio.load_yaml(path)["left"]

    assert rio.get_boxes(shot, "S") == [("1", [5.0, 5.0, 15.0, 15.0]), ("2", [20.0, 5.0, 30.0, 15.0])]
    assert rio.get_boxes(shot, "H") == [("1", [30.0, 30.0, 40.0, 40.0])]

    rio.set_boxes(shot, "S", [])
    assert "S" not in shot and "H" in shot
