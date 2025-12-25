# app/core/worker.py

import threading
from typing import Callable, Dict, Any

from app.core.job_manager import JobManager
from app.utils.pdf_reader import extract_text_from_pdf


class AsyncWorker:
    """
    Fake async worker using background thread.
    Updated for 3-stage evaluation pipeline.
    """

    def __init__(self, job_manager: JobManager):
        self.job_manager = job_manager

    def run_job(
        self,
        job_id: str,
        cv_pdf_path: str,
        project_pdf_path: str,  # CHANGED: job_pdf_path → project_pdf_path
        task_fn: Callable[[str, str, str], Dict[str, Any]],  # CHANGED: takes 3 params
    ):
        """
        Start background job for evaluation.
        
        Args:
            job_id: Unique job identifier
            cv_pdf_path: Path to candidate CV PDF
            project_pdf_path: Path to project report PDF (not job description)
            task_fn: Function that takes (cv_text, project_text, job_title) → result dict
        """
        thread = threading.Thread(
            target=self._execute,
            args=(job_id, cv_pdf_path, project_pdf_path, task_fn),
            daemon=True
        )
        thread.start()

    def _execute(
        self,
        job_id: str,
        cv_pdf_path: str,
        project_pdf_path: str,  # CHANGED: job_pdf_path → project_pdf_path
        task_fn: Callable[[str, str, str], Dict[str, Any]],  # CHANGED: takes 3 params
    ):
        """
        Execute the evaluation pipeline in background.
        """
        try:
            self.job_manager.set_processing(job_id)

            # Read PDFs
            cv_text = extract_text_from_pdf(cv_pdf_path)
            project_text = extract_text_from_pdf(project_pdf_path)  # CHANGED

            # Default job title (can be extended to pass as parameter)
            job_title = "Backend Developer"
            
            # Run 3-stage evaluation pipeline
            # task_fn now expects: (cv_text, project_text, job_title)
            result = task_fn(cv_text, project_text, job_title)

            self.job_manager.set_completed(job_id, result)

        except Exception as e:
            self.job_manager.set_failed(job_id, str(e))
            # Log the error for debugging
            print(f"❌ Job {job_id} failed: {e}")
