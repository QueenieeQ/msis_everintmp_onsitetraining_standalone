# mixins/output_lifecycle.py
# ------------------------------------------------------------
# _OutputLifecycleMixin: archives a pre-existing output folder before a
# fresh registration run, and writes the per-run registration_info.txt
# timing summary. Split out of ReferenceDataManager (model.py) to keep
# files under 300 lines.
# ------------------------------------------------------------

import datetime
import os
import shutil
from pathlib import Path

from onsite_training_studio.colors import green, yellow


class _OutputLifecycleMixin:
    def _archive_existing_output_root(self, destination_root):
        """
        If destination_root already exists, rename it with current date/time,
        then move that renamed folder into an old_registration_data archive
        folder next to it, so only the newest successfully trained printer
        stays in the parent folder.

        Example:
            mpback/realsense/XM7-40
        becomes:
            mpback/realsense/old_registration_data/XM7-40-20260712_223015

        Then the new run can create a fresh XM7-40 folder.
        """
        destination_root = Path(destination_root)

        if not destination_root.exists():
            return None

        if not destination_root.is_dir():
            raise NotADirectoryError(
                f"Output path already exists but is not a directory: {destination_root}"
            )

        archive_dir = destination_root.parent / "old_registration_data"
        os.makedirs(archive_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_root = archive_dir / f"{destination_root.name}-{timestamp}"

        # Extra safety if same timestamp already exists
        counter = 1
        while backup_root.exists():
            backup_root = archive_dir / f"{destination_root.name}-{timestamp}_{counter}"
            counter += 1

        shutil.move(str(destination_root), str(backup_root))

        self.logger.info(
            yellow(f"Existing output folder archived: {destination_root} -> {backup_root}")
        )

        return backup_root

    def _archive_existing_shot_root(self, shot_root):
        """Like _archive_existing_output_root but scoped to one shot
        subfolder, for independent per-shot (re)training that must not
        disturb sibling shot folders under the same destination_root.

        Example:
            mpback/realsense/XM7-40/left
        becomes:
            mpback/realsense/old_registration_data/XM7-40_left-20260712_223015
        """
        shot_root = Path(shot_root)

        if not shot_root.exists():
            return None

        if not shot_root.is_dir():
            raise NotADirectoryError(
                f"Shot output path already exists but is not a directory: {shot_root}"
            )

        archive_dir = shot_root.parent.parent / "old_registration_data"
        os.makedirs(archive_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{shot_root.parent.name}_{shot_root.name}"
        backup_root = archive_dir / f"{stem}-{timestamp}"

        counter = 1
        while backup_root.exists():
            backup_root = archive_dir / f"{stem}-{timestamp}_{counter}"
            counter += 1

        shutil.move(str(shot_root), str(backup_root))

        self.logger.info(
            yellow(f"Existing shot output folder archived: {shot_root} -> {backup_root}")
        )

        return backup_root

    def _format_elapsed_time(self, start_time, end_time):
        """Format elapsed time between two datetime objects as HH:MM:SS."""
        total_seconds = int((end_time - start_time).total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _write_registration_info(self, destination_root, start_time, end_time):
        """
        Write registration timing information inside the printer_id output folder.

        Example output file:
            <destination_root>/registration_info.txt
        """
        destination_root = Path(destination_root)
        destination_root.mkdir(parents=True, exist_ok=True)

        info_path = destination_root / "registration_info.txt"
        total_time = self._format_elapsed_time(start_time, end_time)

        lines = [
            f"Registration Date : {start_time.strftime('%Y-%m-%d')}",
            f"Registration Time : {start_time.strftime('%H:%M:%S')}",
            f"Starting Time : {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Ending Time : {end_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Time Taken for Training : {total_time}",
        ]

        info_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.logger.info(green(f"Registration info saved: {info_path}"))
        return info_path
