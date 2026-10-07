# mixins/registration_context.py
# ------------------------------------------------------------
# Reads a registration YAML and infers (product_root, printer_id, camera,
# side) from its path:
#   .../Product_Registration_Information/<mpfront|mpback>/<camera>/<printer_id>/configs/<file>.yaml
# ------------------------------------------------------------

from pathlib import Path

import yaml


class _RegistrationContextMixin:
    def _load_yaml_file(self, yaml_file_path):
        yaml_file_path = Path(yaml_file_path)
        if not yaml_file_path.exists():
            raise FileNotFoundError(f"Registration YAML file not found: {yaml_file_path}")
        with open(yaml_file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _infer_input_context_from_yaml_path(self, yaml_file_path):
        """Return (product_root, printer_id, camera_folder, side_folder)."""
        yaml_file_path = Path(yaml_file_path).resolve()
        product_root = yaml_file_path.parent.parent
        printer_id = product_root.name
        camera_folder = product_root.parent.name.lower()

        path_parts_lower = [p.lower() for p in yaml_file_path.parts]
        yaml_name_lower = yaml_file_path.name.lower()

        # Prefer the YAML filename when it clearly says front/back.
        if "front" in yaml_name_lower:
            side_folder = "mpfront"
        elif "back" in yaml_name_lower:
            side_folder = "mpback"
        elif "mpfront" in path_parts_lower:
            side_folder = "mpfront"
        elif "mpback" in path_parts_lower:
            side_folder = "mpback"
        else:
            raise ValueError(
                f"Could not infer side folder from YAML path: {yaml_file_path}. "
                "Expected mpfront/mpback in path or front/back in YAML filename."
            )

        if camera_folder not in ["realsense", "omron", "top_camera"]:
            raise ValueError(
                f"Could not infer camera folder from YAML path: {yaml_file_path}. "
                "Expected product folder under realsense or omron."
            )

        return product_root, printer_id, camera_folder, side_folder
