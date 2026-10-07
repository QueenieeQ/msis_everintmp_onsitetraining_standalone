# screw_head.py
# ------------------------------------------------------------
# Screw-head (label "S") constants and annotation helpers for onsite
# training. S is trained as its OWN model, separate from the screw-hole (H)
# model that model_1 loads from weights/weight.pt:
#
#   <shot>/labels/ref_user_screw_head.txt            YOLO-ish "0 x1 y1 x2 y2"
#   <shot>/patches/screw_head/ref_user_S_<i>_<area>.png
#   <shot>/weights/screw_head/screw_head_detection.pt
# ------------------------------------------------------------

import cv2

SCREW_HEAD_LABEL = "S"
SCREW_HEAD_CLASS_NAME = "screw_head"
SCREW_HEAD_LABELS_FILE = "ref_user_screw_head.txt"
SCREW_HEAD_PATCH_DIR = "patches/screw_head"
SCREW_HEAD_WEIGHTS_SUBDIR = "screw_head"
SCREW_HEAD_WEIGHTS_FILENAME = "screw_head_detection.pt"


def collect_screw_head_annotations(manager, shot_root, shot_name, shot_data, image, log_warn):
    """Build S annotations + label lines and write their patches.

    Returns ``(annotations, label_lines)``; empty when the shot has no S.
    Bad entries are skipped with a warning, never raised (same policy as H).
    """
    annotations, label_lines = [], []
    nodes = shot_data.get(SCREW_HEAD_LABEL)
    if not isinstance(nodes, dict):
        return annotations, label_lines

    index = 0
    for _, node in sorted(nodes.items(), key=lambda item: str(item[0])):
        try:
            x1, y1, x2, y2 = manager._clip_bbox_to_image(manager._extract_bbox(node), image.shape)
            patch = image[y1:y2, x1:x2]
            if patch.size == 0:
                log_warn(f"Empty S patch skipped for {shot_name}: {[x1, y1, x2, y2]}")
                continue
            area = int(patch.shape[0] * patch.shape[1])
            patch_abs = shot_root / SCREW_HEAD_PATCH_DIR / f"ref_user_S_{index}_{area}.png"
            patch_abs.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(patch_abs), patch)
            label_lines.append(f"0 {x1} {y1} {x2} {y2}")
            annotations.append({
                "label": SCREW_HEAD_LABEL,
                "bbox": [x1, y1, x2, y2],
                "location_img_patch": manager._model_relative_path(patch_abs),
            })
            index += 1
        except Exception as exc:
            log_warn(f"S annotation skipped for {shot_name}: {exc}")
    return annotations, label_lines
