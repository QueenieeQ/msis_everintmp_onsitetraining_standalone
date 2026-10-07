"""Onsite-registration error codes (115-xxx).

Numbers match the vision framework's; labels here follow how each suffix is
actually raised (the framework registry's 003/005 labels do not match its own
call sites).
"""

from __future__ import annotations

_GROUPS = {
    "onsite_registration": (
        115,
        {
            0: "success",
            1: "registration_yaml_missing",
            2: "invalid_registration_yaml",
            3: "reserved",
            4: "missing_registration_images",
            5: "onsite_training_failed",
            6: "invalid_information_path",
            7: "reference_folder_structure_failed",
        },
    ),
}


def error_code_str(group: str, suffix: int) -> str:
    return f"{_GROUPS[group][0]:03d}-{int(suffix):03d}"


def error_label(group: str, suffix: int) -> str:
    return _GROUPS[group][1].get(int(suffix), f"{group}_{int(suffix):03d}")


def error_comment(group: str, suffix: int, detail: str = "") -> str:
    label = error_label(group, suffix)
    detail = str(detail or "").strip()
    return label if not detail else f"{label}: {detail}"
