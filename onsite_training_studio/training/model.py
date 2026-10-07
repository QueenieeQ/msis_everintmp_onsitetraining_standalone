# model.py
# ------------------------------------------------------------
# ReferenceDataManager: composition root for the mixins under mixins/.
# Public entry points (build_onsite_training_data / infer / process) live in
# builder_api.py and information_path.py and are re-exported here.
# ------------------------------------------------------------

import os

from loguru import logger

from onsite_training_studio.colors import green
from onsite_training_studio.training.mixins.output_lifecycle import _OutputLifecycleMixin
from onsite_training_studio.training.mixins.reference_tree_builder import _ReferenceTreeBuilderMixin
from onsite_training_studio.training.mixins.registration_context import _RegistrationContextMixin
from onsite_training_studio.training.mixins.shot_parsing import _ShotParsingMixin


class ReferenceDataManager(
    _RegistrationContextMixin,
    _ShotParsingMixin,
    _OutputLifecycleMixin,
    _ReferenceTreeBuilderMixin,
):
    """Builds a product's per-shot reference tree (images, labels, patches,
    config) from its registration YAML and trains the selected models."""

    def __init__(self, reference_folder=None, roi_extend=20, create_root_dirs=False):
        self.logger = logger
        self.reference_folder = reference_folder
        self.source_folder = reference_folder
        self.extract_roi_extend = roi_extend
        self._rel_prefix = ""
        os.makedirs(self.reference_folder, exist_ok=True)
        self.logger.info(green(f"Reference folder initialized: {self.reference_folder}"))


from onsite_training_studio.training.builder_api import (  # noqa: E402
    OnsiteTrainingDataBuilder,
    build_onsite_training_data,
)
from onsite_training_studio.training.information_path import (  # noqa: E402
    infer,
    parse_information_path,
    process,
    resolve_registration_yaml,
)

__all__ = [
    "OnsiteTrainingDataBuilder",
    "ReferenceDataManager",
    "build_onsite_training_data",
    "infer",
    "parse_information_path",
    "process",
    "resolve_registration_yaml",
]
