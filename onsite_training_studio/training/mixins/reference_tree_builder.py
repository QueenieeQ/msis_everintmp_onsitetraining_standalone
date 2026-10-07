# mixins/reference_tree_builder.py
# ------------------------------------------------------------
# _ReferenceTreeBuilderMixin: the current/recommended registration pipeline
# — reads a printer's registration YAML (path-derived printer_id/camera/
# side context), builds its per-shot reference-data tree (images, labels,
# patches, config), and optionally triggers one-shot training. Split out of
# ReferenceDataManager (model.py) to keep files under 300 lines.
# ------------------------------------------------------------

import datetime
import json
import os
import traceback
from pathlib import Path

import cv2
import numpy as np
import yaml

from onsite_training_studio.colors import green, red, yellow
from onsite_training_studio.training.information_path import OnsiteRegistrationError
from onsite_training_studio.paths import output_root, resolve_workspace_path
from onsite_training_studio.training.model_selection import (
    normalize_model_variant,
    normalize_only_models,
    train_selected_models,
)
from onsite_training_studio.training.screw_head import (
    SCREW_HEAD_LABELS_FILE,
    SCREW_HEAD_WEIGHTS_FILENAME,
    SCREW_HEAD_WEIGHTS_SUBDIR,
    collect_screw_head_annotations,
)
from onsite_training_studio.training.shot_selection import filter_shots_for_build


def _shots_config_from_registration(parsed_shots, printer_id, camera_folder, side_folder):
    """Derive a model_1 mp_vision_params `shots:` block from registration data.

    `parsed_shots` is `_iter_registration_shots(data)`'s output: (shot_name,
    shot_data) pairs straight from the UI-provided registration_info.yaml
    (e.g. `roi`/`closer`/`left`/`right`, each with an "H" bbox-per-hole dict
    and/or a "P" ROI bbox). Mirrors the shot_id convention model_1 already
    parses elsewhere (`<camera>_<front|back>_<position>` — see
    utils.parse_camera/parse_surface): shots with a "P" key (or literally
    named "roi") become shot_type=roi_detection with expected_holes=0 always
    — ROI shots verify a bounding box, not a hole count. Every other shot is
    shot_type=screw_detection, with expected_holes = number of "H" entries
    when present, or 0 when the shot has no "H" annotations yet (image only,
    nothing annotated) — kept in the shots: block rather than dropped, so an
    MQTT command still targeting that shot_id gets a graceful "0 expected
    holes, pass" result instead of an "Unknown shot_id" error.
    """
    side_token = side_folder[2:]  # "mpfront" -> "front", "mpback" -> "back"
    shots_config = {}
    for shot_name, shot_data in parsed_shots:
        if "P" in shot_data or shot_name == "roi":
            shot_type = "roi_detection"
            expected_holes = 0
        else:
            shot_type = "screw_detection"
            holes = shot_data.get("H")
            expected_holes = len(holes) if isinstance(holes, dict) else 0

        shot_id = f"{camera_folder}_{side_token}_{shot_name}"
        shots_config[shot_id] = {
            "shot_type": shot_type,
            "expected_holes": expected_holes,
            "ref_json_subpath": (
                f"{side_folder}/{camera_folder}/{printer_id}/{shot_name}/"
                "config/ref_user_extracted.json"
            ),
        }
    return shots_config


def _write_shots_summary(destination_root, shots_config):
    """Write the per-shot summary (shot_type / expected_holes / ref json path).

    Standalone replacement for model_1's ``mp_vision_params`` scaffold: copy the
    ``shots:`` block into that printer's model_1 config on the vision PC.
    """
    path = Path(destination_root) / "shots_config.yaml"
    path.write_text(yaml.safe_dump({"shots": shots_config}, sort_keys=False), encoding="utf-8")
    return path


def _load_trainers():
    """Import trainers lazily — torch/ultralytics are only needed when training."""
    from onsite_training_studio.training.training import train_roi_one_shot
    from onsite_training_studio.training.training_patch import train_patch_one_shot
    from onsite_training_studio.training.training_screw import train_screw_one_shot
    from onsite_training_studio.training.training_screw_head import train_screw_head_one_shot

    return {
        "roi": train_roi_one_shot,
        "screw_hole": train_screw_one_shot,
        "patches": train_patch_one_shot,
        "screw_head": train_screw_head_one_shot,
    }


# Trained output keeps model_1's relative-path convention inside
# ref_user_extracted.json, so a finished product folder can be copied straight
# into vision_system_fw/data/models/model_1/<stage>/<camera>/<product>/.
MODEL_1_REL_PREFIX = "data/models/model_1"


class _ReferenceTreeBuilderMixin:
    def build_reference_tree_from_registration_yaml_file(
        self, yaml_file_path, destination_root=None, run_training=True, only_shots=None,
        only_models=None, model_variant=None,
    ):
        """
        New recommended builder.

        Minimal input:
            yaml_file_path = .../Product_Registration_Information/mpback/realsense/XM7-40/configs/XM7-40_back_registration_info.yaml

        Output (default <workspace>/output/..., laid out like model_1's data dir):
            <workspace>/output/mpback/realsense/XM7-40/<shot>/
                config/
                images/
                labels/
                patches/
                weights/

        Notes:
            - Shot names are read directly from YAML and kept lowercase.
            - No fixed assumption about left/right/closer/closer/roi.
            - Any shot with H is treated as hole/screw training data.
            - Any shot with P is treated as ROI training data.
            - `only_shots` (optional iterable of shot names, e.g. ["left"]):
              build/train only those shots instead of every shot in the YAML.
              Unselected shots need no image and are left untouched on disk —
              the printer's persisted `shots:` config is still derived from
              *all* shots in the YAML (not just the selected ones), so a
              partial build never wipes another shot's expected_holes entry.
            - `only_models` (optional iterable of screw_hole|patches|screw_head|roi):
              train just those model kinds for the selected shots. Existing
              output is updated in place (NOT archived) so the other models'
              weights survive. Default None = every model the shot has data for.
            - `model_variant` (optional, e.g. "yolo11m.pt"): YOLO weights to
              fine-tune from, overriding the onsite YAML's train.model_variant.
        """
        registration_start_time = datetime.datetime.now()
        only_models = normalize_only_models(only_models)
        model_variant = normalize_model_variant(model_variant)

        yaml_file_path = resolve_workspace_path(yaml_file_path)

        try:
            data = self._load_yaml_file(yaml_file_path)
        except Exception as exc:
            raise OnsiteRegistrationError(2, f"Invalid registration yaml {yaml_file_path}: {exc}") from exc

        try:
            product_root, printer_id, camera_folder, side_folder = self._infer_input_context_from_yaml_path(yaml_file_path)
        except Exception as exc:
            raise OnsiteRegistrationError(6, f"Invalid registration path {yaml_file_path}: {exc}") from exc

        all_shots = self._iter_registration_shots(data)
        if not all_shots:
            raise OnsiteRegistrationError(2, f"No valid shots found in YAML: {yaml_file_path}")
        shots = filter_shots_for_build(all_shots, only_shots)

        if destination_root:
            destination_root = resolve_workspace_path(destination_root)
        else:
            destination_root = output_root() / side_folder / camera_folder / printer_id

        try:
            if only_models:
                # Model-subset build: keep everything already on disk and
                # refresh in place so untouched models' weights survive.
                os.makedirs(destination_root, exist_ok=True)
            elif only_shots:
                # Partial (single/subset-shot) build: never touch sibling shot
                # folders under destination_root — only reset the ones we're
                # about to rebuild.
                os.makedirs(destination_root, exist_ok=True)
                for shot_name, _ in shots:
                    self._archive_existing_shot_root(destination_root / shot_name)
            else:
                # Full build: if output folder already exists, rename the old
                # one first, then create a fresh output folder.
                self._archive_existing_output_root(destination_root)
                os.makedirs(destination_root, exist_ok=True)
        except OSError as exc:
            raise OnsiteRegistrationError(7, f"Failed to prepare output folder {destination_root}: {exc}") from exc

        self.reference_folder = str(destination_root)
        self._rel_prefix = f"{MODEL_1_REL_PREFIX}/{side_folder}/{camera_folder}/{printer_id}"
        model_1_shots = _shots_config_from_registration(all_shots, printer_id, camera_folder, side_folder)

        self.logger.info(green(f"Input YAML: {yaml_file_path}"))
        self.logger.info(green(f"Product root: {product_root}"))
        self.logger.info(green(f"Output root: {destination_root}"))
        self.logger.info(green(f"Detected shots: {[name for name, _ in shots]}"))

        trainers = {}
        if run_training:
            try:
                trainers = _load_trainers()
            except Exception as load_exc:
                self.logger.error(red(f"Training functions could not be loaded: {load_exc}"))
                self.logger.error(red(traceback.format_exc()))

        missing_image_shots = []
        training_failed_shots = []
        shot_structure_errors = []

        for shot_name, shot_data in shots:
            try:
                shot_root = destination_root / shot_name
                self._ensure_shot_dirs(str(shot_root))

                source_image_path = self._resolve_source_image_path_from_yaml_file(product_root, shot_name, shot_data)
                source_image_path = Path(source_image_path)
                if not source_image_path.exists():
                    self.logger.warning(yellow(f"Missing source image for shot '{shot_name}': {source_image_path}"))
                    missing_image_shots.append(shot_name)
                    continue

                destination_image_path = shot_root / "images" / "ref_user.png"
                self._copy_image(str(source_image_path), str(destination_image_path))

                image = cv2.imread(str(source_image_path))
                if image is None:
                    self.logger.warning(yellow(f"Unable to read source image for shot '{shot_name}': {source_image_path}"))
                    missing_image_shots.append(shot_name)
                    continue

                image_h, image_w = image.shape[:2]
                annotations = []
                label_lines = []
                roi_label_lines = []

                image_rel_path = self._model_relative_path(shot_root / "images" / "ref_user.png")
                mask_rel_path = self._model_relative_path(shot_root / "patches" / "mask_ref" / "mask_ref_user.png")
                weight_rel_path = self._model_relative_path(shot_root / "weights" / "weight.pt")

                if "H" in shot_data and isinstance(shot_data["H"], dict):
                    hole_index = 0
                    for _, hole_node in sorted(shot_data["H"].items(), key=lambda item: str(item[0])):
                        try:
                            bbox = self._extract_bbox(hole_node)
                            x1, y1, x2, y2 = self._clip_bbox_to_image(bbox, image.shape)
                            patch = image[y1:y2, x1:x2]
                            if patch.size == 0:
                                self.logger.warning(yellow(f"Empty H patch skipped for {shot_name}: {[x1, y1, x2, y2]}"))
                                continue
                            patch_area = int(patch.shape[0] * patch.shape[1])
                            patch_abs_path = shot_root / "patches" / "hole" / f"ref_user_H_{hole_index}_{patch_area}.png"
                            cv2.imwrite(str(patch_abs_path), patch)
                            patch_rel_path = self._model_relative_path(patch_abs_path)
                            label_lines.append(f"0 {x1} {y1} {x2} {y2}")
                            annotations.append({
                                "label": "H",
                                "bbox": [x1, y1, x2, y2],
                                "location_img_patch": patch_rel_path,
                            })
                            hole_index += 1
                        except Exception as h_exc:
                            self.logger.warning(yellow(f"H annotation skipped for {shot_name}: {h_exc}"))

                if "P" in shot_data:
                    try:
                        p_bbox = self._extract_bbox(shot_data["P"])
                        x1, y1, x2, y2 = self._clip_bbox_to_image(p_bbox, image.shape)
                        roi_crop = image[y1:y2, x1:x2]
                        if roi_crop.size == 0:
                            self.logger.warning(yellow(f"Empty P/ROI patch skipped for {shot_name}: {[x1, y1, x2, y2]}"))
                        else:
                            roi_area = int(roi_crop.shape[0] * roi_crop.shape[1])
                            roi_abs_path = shot_root / "patches" / "roi" / f"ref_user_roi_{roi_area}.png"
                            cv2.imwrite(str(roi_abs_path), roi_crop)
                            roi_rel_path = self._model_relative_path(roi_abs_path)
                            roi_label_lines.append(f"0 {' '.join(str(v) for v in self._bbox_to_polygon([x1, y1, x2, y2]))}")
                            annotations.append({
                                "label": "P",
                                "bbox": [x1, y1, x2, y2],
                                "location_roi": roi_rel_path,
                            })
                    except Exception as p_exc:
                        self.logger.warning(yellow(f"P/ROI annotation skipped for {shot_name}: {p_exc}"))

                # S = screw head: its own labels file/patches/model, never part of the H hole mask.
                s_annotations, s_label_lines = collect_screw_head_annotations(
                    self, shot_root, shot_name, shot_data, image,
                    lambda msg: self.logger.warning(yellow(msg)),
                )
                annotations.extend(s_annotations)

                mask_image = np.zeros((image_h, image_w), dtype=np.uint8)
                for entry in annotations:
                    if entry.get("label") == "H":
                        x1, y1, x2, y2 = entry["bbox"]
                        mask_image[y1:y2, x1:x2] = 255

                mask_abs_path = shot_root / "patches" / "mask_ref" / "mask_ref_user.png"
                cv2.imwrite(str(mask_abs_path), mask_image)

                self._write_lines(str(shot_root / "labels" / "ref_user.txt"), label_lines)
                self._write_lines(str(shot_root / "labels" / "ref_user_roi.txt"), roi_label_lines)
                if s_label_lines:
                    self._write_lines(str(shot_root / "labels" / SCREW_HEAD_LABELS_FILE), s_label_lines)

                extracted_cfg = {
                    "image": image_rel_path,
                    "mask_path": mask_rel_path,
                    "weight_path": weight_rel_path,
                    "annotations": annotations,
                }

                if s_annotations:
                    extracted_cfg["screw_head_weight_path"] = self._model_relative_path(
                        shot_root / "weights" / SCREW_HEAD_WEIGHTS_SUBDIR / SCREW_HEAD_WEIGHTS_FILENAME
                    )

                cfg_path = shot_root / "config" / "ref_user_extracted.json"
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump(extracted_cfg, f, indent=4)

                if run_training:
                    try:
                        shot_image_path = shot_root / "images" / "ref_user.png"
                        where = f"{side_folder}/{camera_folder}/{printer_id}/{shot_name}"
                        ref_root = str(destination_root)
                        p_ann = next((a for a in annotations if a.get("label") == "P"), None)
                        p_bbox = (p_ann["bbox"] if p_ann else self._extract_bbox(shot_data["P"])) if "P" in shot_data else None
                        train_selected_models(
                            only_models=only_models,
                            model_variant=model_variant,
                            has_data={
                                "screw_hole": "H" in shot_data,
                                "patches": "H" in shot_data,
                                "screw_head": bool(s_label_lines),
                                "roi": "P" in shot_data,
                            },
                            trainers=trainers,
                            calls={
                                "screw_hole": lambda t, kw: t(printer_name=shot_name, ref_data_root=ref_root, **kw),
                                "patches": lambda t, kw: t(printer_name=shot_name, ref_data_root=ref_root, **kw),
                                "screw_head": lambda t, kw: t(printer_name=shot_name, ref_data_root=ref_root, **kw),
                                "roi": lambda t, kw: t(str(shot_image_path), tuple(p_bbox), shot_name, **kw),
                            },
                            on_start=lambda kind: self.logger.info(green(f"Training {kind} model for {where}")),
                        )
                    except Exception as train_exc:
                        self.logger.error(red(f"Training failed for {shot_name}: {train_exc}"))
                        self.logger.error(red(traceback.format_exc()))
                        training_failed_shots.append(shot_name)

                self.logger.info(green(f"Created reference tree for {shot_name}: {cfg_path}"))

            except Exception as shot_exc:
                self.logger.error(red(f"Shot failed and was skipped: {shot_name}: {shot_exc}"))
                self.logger.error(red(traceback.format_exc()))
                shot_structure_errors.append(shot_name)
                continue

        _write_shots_summary(destination_root, model_1_shots)
        registration_end_time = datetime.datetime.now()
        self._write_registration_info(destination_root, registration_start_time, registration_end_time)

        if missing_image_shots:
            raise OnsiteRegistrationError(4, f"Missing reference images for shots: {missing_image_shots}")
        if shot_structure_errors:
            raise OnsiteRegistrationError(7, f"Reference folder structure failed for shots: {shot_structure_errors}")
        if run_training and training_failed_shots:
            raise OnsiteRegistrationError(5, f"Onsite training failed for shots: {training_failed_shots}")

        return str(destination_root)
