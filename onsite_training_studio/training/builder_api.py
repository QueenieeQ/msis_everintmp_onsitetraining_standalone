# builder_api.py
# ------------------------------------------------------------
# OnsiteTrainingDataBuilder / build_onsite_training_data: the minimal
# outside-call API for building a printer's reference-data tree (and
# optionally training) from a registration YAML path. Split out of
# model.py to keep files under 300 lines.
#
# ReferenceDataManager is imported lazily (inside the function body) to
# avoid a circular import: model.py imports OnsiteTrainingDataBuilder /
# build_onsite_training_data from this module for re-export, so this
# module can't import ReferenceDataManager from model.py at module load
# time. Same pattern information_path.py already uses to call back into
# model.py.
# ------------------------------------------------------------

from onsite_training_studio.paths import resolve_workspace_path


class OnsiteTrainingDataBuilder:
    """
    Simple callable wrapper for external integration.

    Usage from another file:
        model = OnsiteTrainingDataBuilder(run_training=True)
        out_root = model(registration_yaml_path)

    The only required input during the call is registration_yaml_path.
    The output path is inferred automatically from the YAML path unless
    destination_root is provided during initialization.
    """

    def __init__(
        self, destination_root=None, run_training=True, roi_extend=0, only_shots=None,
        only_models=None, model_variant=None,
    ):
        self.destination_root = destination_root
        self.run_training = run_training
        self.roi_extend = roi_extend
        self.only_shots = only_shots
        self.only_models = only_models
        self.model_variant = model_variant

    def __call__(self, registration_yaml_path):
        return build_onsite_training_data(
            registration_yaml_path=registration_yaml_path,
            destination_root=self.destination_root,
            run_training=self.run_training,
            roi_extend=self.roi_extend,
            only_shots=self.only_shots,
            only_models=self.only_models,
            model_variant=self.model_variant,
        )

    def build(self, registration_yaml_path):
        return self(registration_yaml_path)


def build_onsite_training_data(
    registration_yaml_path, destination_root=None, run_training=True, roi_extend=0, only_shots=None,
    only_models=None, model_variant=None,
):
    """
    Minimal outside-call wrapper.

    Required:
        registration_yaml_path

    Optional:
        destination_root: if None, <workspace>/output/<stage>/<camera>/<printer_id>.
        run_training: True to also run one-shot training, False to only create files/folders
        roi_extend: kept for compatibility
        only_shots: iterable of shot names (e.g. ["left"]) to build/train
            independently of the others; None (default) builds every shot
            in the YAML, unchanged from prior behavior.
        only_models: iterable of screw_hole|patches|screw_head|roi to train just
            those models (existing output updated in place, not archived);
            None trains every model the shot has annotations for.
        model_variant: YOLO weights name (e.g. "yolo11m.pt") overriding the
            onsite YAML's train.model_variant; None keeps the config value.
    """
    from onsite_training_studio.training.model import ReferenceDataManager

    registration_yaml_path = resolve_workspace_path(registration_yaml_path)
    destination_root = resolve_workspace_path(destination_root) if destination_root else None

    manager = ReferenceDataManager(
        reference_folder=str(destination_root) if destination_root else str(registration_yaml_path.parent.parent),
        roi_extend=roi_extend,
        create_root_dirs=False,
    )
    return manager.build_reference_tree_from_registration_yaml_file(
        yaml_file_path=registration_yaml_path,
        destination_root=destination_root,
        run_training=run_training,
        only_shots=only_shots,
        only_models=only_models,
        model_variant=model_variant,
    )
