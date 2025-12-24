# app/core/job_manager.py

import uuid
from datetime import datetime
from typing import Dict, Optional


class JobStatus:
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobManager:
    """
    In-memory job manager.
    Responsible ONLY for job lifecycle and state.
    """

    def __init__(self):
        self._jobs: Dict[str, Dict] = {}

    def create_job(self) -> str:
        job_id = str(uuid.uuid4())

        self._jobs[job_id] = {
            "job_id": job_id,
            "status": JobStatus.QUEUED,
            "result": None,
            "error": None,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }

        return job_id

    def set_processing(self, job_id: str):
        self._update_job(job_id, status=JobStatus.PROCESSING)

    def set_completed(self, job_id: str, result: Dict):
        self._update_job(
            job_id,
            status=JobStatus.COMPLETED,
            result=result,
            error=None
        )

    def set_failed(self, job_id: str, error: str):
        self._update_job(
            job_id,
            status=JobStatus.FAILED,
            result=None,
            error=error
        )

    def get_job(self, job_id: str) -> Optional[Dict]:
        return self._jobs.get(job_id)

    def _update_job(
        self,
        job_id: str,
        status: str,
        result: Optional[Dict] = None,
        error: Optional[str] = None,
    ):
        if job_id not in self._jobs:
            raise ValueError(f"Job {job_id} not found")

        self._jobs[job_id]["status"] = status
        self._jobs[job_id]["result"] = result
        self._jobs[job_id]["error"] = error
        self._jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()
