# mixins/shot_parsing.py
# ------------------------------------------------------------
# _ShotParsingMixin: per-shot geometry/path helpers and small file-I/O
# primitives (write label lines, copy an image, save a crop) used while
# building a printer's reference-data tree. Split out of ReferenceDataManager
# (model.py) to keep files under 300 lines.
# ------------------------------------------------------------

import os
import shutil
from pathlib import Path

import cv2


class _ShotParsingMixin:
    def _resolve_source_image_path(self, source_root, shot_name, shot_data):
        """
        Resolve source image path safely for both old and new structures.

        Supports YAML image_path formats such as:
            images/ROI.png
            FRONT/images/ROI.png
            BACK/images/CENTER.png
            FRONT\\images\\LEFT.png

        Also falls back to:
            source_root/images/<shot_name>.png
        """
        source_root = Path(source_root)
        image_path_from_yaml = shot_data.get("image_path") if isinstance(shot_data, dict) else None

        if image_path_from_yaml:
            image_path_from_yaml = str(image_path_from_yaml).replace("\\", os.sep).replace("/", os.sep)
            image_path_obj = Path(image_path_from_yaml)

            # Absolute image path in YAML.
            if image_path_obj.is_absolute():
                return image_path_obj

            image_parts = image_path_obj.parts

            # If YAML says FRONT/images/ROI.png or BACK/images/ROI.png,
            # remove FRONT/BACK because source_root already points to the product/side folder.
            if len(image_parts) >= 2 and image_parts[0].upper() in ["FRONT", "BACK"]:
                return source_root / Path(*image_parts[1:])

            # If YAML says images/ROI.png, use source_root/images/ROI.png.
            return source_root / image_path_obj

        # Fallback when image_path is missing.
        return source_root / "images" / f"{shot_name}.png"

    def _clip_bbox_to_image(self, bbox, image_shape):
        """Clip bbox to image boundaries and return a valid x1,y1,x2,y2 tuple."""
        h, w = image_shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        return x1, y1, x2, y2

    def _ensure_shot_dirs(self, shot_root):
        for rel_path in (
            "config",
            "images",
            "labels",
            "weights",
            "patches/roi",
            "patches/hole",
            "patches/mask_ref",
        ):
            os.makedirs(os.path.join(shot_root, rel_path), exist_ok=True)

    def _extract_bbox(self, node):
        if isinstance(node, dict):
            if "bbox" in node and isinstance(node["bbox"], list):
                return node["bbox"]
            for value in node.values():
                if isinstance(value, dict) and "bbox" in value:
                    return value["bbox"]
        if isinstance(node, list):
            return node
        raise ValueError(f"Unable to extract bbox from node: {node}")

    def _bbox_to_polygon(self, bbox):
        x1, y1, x2, y2 = bbox
        return [x1, y1, x2, y1, x2, y2, x1, y2]

    def _write_lines(self, file_path, lines):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")

    def _copy_image(self, source_path, destination_path):
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.copy2(source_path, destination_path)

    def _save_crop(self, image, bbox, output_path):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = max(x1 + 1, x2)
        y2 = max(y1 + 1, y2)
        crop = image[y1:y2, x1:x2]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, crop)
        return crop

    def _normalize_shot_data_keys(self, shot_data):
        """Return a copy of shot_data where annotation keys H/P/S are normalized to uppercase."""
        if not isinstance(shot_data, dict):
            return {}

        normalized = {}
        for key, value in shot_data.items():
            if isinstance(key, str) and key.upper() in ["H", "P", "R", "S", "X", "Y"]:
                normalized[key.upper()] = value
            else:
                normalized[key] = value
        return normalized

    def _iter_registration_shots(self, data):
        """
        Iterate shots from the new YAML format.

        New YAML format is direct shot-level:
            closer:
              image_path: images\\closer.png
              H: ...
            roi:
              image_path: images\\roi.png
              P: ...

        It also tolerates older wrappers such as FRONT/BACK if they appear.
        """
        if not isinstance(data, dict):
            return []

        ignored_keys = {"printer_name", "camera_system", "side", "camera", "printer_id"}

        # If old YAML has FRONT/BACK as a single wrapper, unwrap it.
        for wrapper in ("front", "back", "FRONT", "BACK"):
            if wrapper in data and isinstance(data[wrapper], dict):
                data = data[wrapper]
                break

        shots = []
        for shot_name, shot_data in data.items():
            if not isinstance(shot_data, dict):
                continue
            if str(shot_name) in ignored_keys or str(shot_name).lower() in ignored_keys:
                continue

            # Accept any shot name, fixed or variable, as long as it looks like a shot block.
            has_image = "image_path" in shot_data
            has_annotation = any(str(k).upper() in ["H", "P", "R", "S", "X", "Y"] for k in shot_data.keys())
            if has_image or has_annotation:
                shots.append((str(shot_name).lower(), self._normalize_shot_data_keys(shot_data)))

        return shots

    def _resolve_source_image_path_from_yaml_file(self, product_root, shot_name, shot_data):
        """
        Resolve source image path for the new input structure.

        New input structure:
            product_root/
              configs/
              images/

        YAML image_path examples:
            images\\closer.png
            images/closer.png
            closer.png
        """
        product_root = Path(product_root)
        image_path_from_yaml = shot_data.get("image_path") if isinstance(shot_data, dict) else None

        if image_path_from_yaml:
            image_path_from_yaml = str(image_path_from_yaml).replace("\\", os.sep).replace("/", os.sep)
            image_path_obj = Path(image_path_from_yaml)

            if image_path_obj.is_absolute():
                return image_path_obj

            image_parts = image_path_obj.parts
            if len(image_parts) >= 1 and image_parts[0].lower() == "images":
                return product_root / image_path_obj

            return product_root / "images" / image_path_obj.name

        return product_root / "images" / f"{shot_name}.png"

    def _model_relative_path(self, abs_path):
        """Path stored in ref_user_extracted.json, in model_1's convention.

        Example (output root = <workspace>/output/mpfront/realsense/XM7-40):
            .../output/mpfront/realsense/XM7-40/left/images/ref_user.png
        becomes:
            data/models/model_1/mpfront/realsense/XM7-40/left/images/ref_user.png
        so the finished folder works once copied into model_1's data dir.
        """
        rel = Path(abs_path).resolve().relative_to(Path(self.reference_folder).resolve())
        return f"{self._rel_prefix}/{rel.as_posix()}"
