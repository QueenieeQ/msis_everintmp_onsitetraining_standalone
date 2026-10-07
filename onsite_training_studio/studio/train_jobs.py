"""Run onsite training (in-process, can take minutes) as a background job.

The HTTP request only starts the job and returns a job id; the browser polls
GET /api/train/{job_id} for progress instead of blocking on one long request.
Training runs are serialized (one at a time) since they share GPU/model
state (GPU).
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

from .train_runner import run_training as run_training_job

_jobs: Dict[str, "TrainJob"] = {}
_jobs_lock = threading.Lock()
_train_lock = threading.Lock()


@dataclass
class TrainJob:
    id: str
    information_path: str
    run_training: bool
    output_folder: Optional[str] = None
    shot: Optional[str] = None
    model: Optional[str] = None
    model_variant: Optional[str] = None
    status: str = "running"
    ok: Optional[bool] = None
    comment: str = ""
    output_root: Optional[str] = None
    error_code: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None


def start_train(
    information_path: str,
    run_training: bool = True,
    output_folder: Optional[str] = None,
    shot: Optional[str] = None,
    model: Optional[str] = None,
    model_variant: Optional[str] = None,
) -> str:
    job = TrainJob(
        id=uuid.uuid4().hex[:12],
        information_path=information_path,
        run_training=run_training,
        output_folder=output_folder or None,
        shot=shot or None,
        model=model or None,
        model_variant=model_variant or None,
    )
    with _jobs_lock:
        _jobs[job.id] = job

    def _run() -> None:
        with _train_lock:
            ok, comment, output_root, error_code = run_training_job(
                information_path, job.run_training, job.output_folder, job.shot,
                job.model, job.model_variant,
            )
        job.ok = ok
        job.comment = comment
        job.output_root = output_root
        job.error_code = error_code
        job.status = "done"
        job.finished_at = time.time()

    threading.Thread(target=_run, daemon=True).start()
    return job.id


def get_job(job_id: str) -> Optional[TrainJob]:
    with _jobs_lock:
        return _jobs.get(job_id)
