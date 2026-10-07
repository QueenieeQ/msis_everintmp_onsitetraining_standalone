"""Filter a registration YAML's parsed shots down to an optional subset.

Split out of mixins/reference_tree_builder.py (already >200 lines) to keep
the `only_shots` addition a focused extraction rather than more growth in
that file.
"""

from __future__ import annotations

from onsite_training_studio.training.information_path import OnsiteRegistrationError


def filter_shots_for_build(all_shots, only_shots):
    """Return the (shot_name, shot_data) pairs to build.

    `all_shots` is `_iter_registration_shots(data)`'s full output. When
    `only_shots` is falsy, every shot is built (unchanged default
    behavior). Otherwise only shots whose (lowercased) name is in
    `only_shots` are returned; raises OnsiteRegistrationError if none match.
    """
    if not only_shots:
        return all_shots
    wanted = {str(s).strip().lower() for s in only_shots}
    shots = [s for s in all_shots if s[0] in wanted]
    if not shots:
        raise OnsiteRegistrationError(
            2,
            f"No shots matched only_shots={sorted(wanted)}; "
            f"available: {[name for name, _ in all_shots]}",
        )
    return shots
