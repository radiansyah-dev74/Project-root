# app/core/worker.py

import threading
from typing import Callable, Dict

from app.core.job_manager import JobManager
from app.utils.pdf_reader import extract_text_from_pdf


class AsyncWorker:
    """
    Fake async worker using background thread.
    """

    def __init__(self, job_manager: JobManager):
        self.job_manager = job_manager

    def run_job(
        self,
        job_id: str,
        cv_pdf_path: str,
        job_pdf_path: str,
        task_fn: Callable[[str, str], Dict],
    ):
        thread = threading.Thread(
            target=self._execute,
            args=(job_id, cv_pdf_path, job_pdf_path, task_fn),
            daemon=True
        )
        thread.start()

    def _execute(
        self,
        job_id: str,
        cv_pdf_path: str,
        job_pdf_path: str,
        task_fn: Callable[[str, str], Dict],
    ):
        try:
            self.job_manager.set_processing(job_id)

            # Read PDFs
            cv_text = extract_text_from_pdf(cv_pdf_path)
            job_text = extract_text_from_pdf(job_pdf_path)

            # Run pipeline (next: RAG)
            result = task_fn(cv_text, job_text)

            self.job_manager.set_completed(job_id, result)

        except Exception as e:
            self.job_manager.set_failed(job_id, str(e))
