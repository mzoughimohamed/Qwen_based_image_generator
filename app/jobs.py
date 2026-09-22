from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass

from app.backends.base import Backend, BackendError
from app.schemas import JobRequest, ValidationError
from app.storage import Storage


@dataclass
class Job:
    id: str
    backend: str
    request: JobRequest | None
    status: str = "queued"
    phase: str | None = None
    step: int | None = None
    total_steps: int | None = None
    result_id: str | None = None
    result: dict | None = None
    rewritten_prompt: str | None = None
    warning: str | None = None
    error: str | None = None


class JobQueue:
    def __init__(self, backends: dict[str, Backend], storage: Storage):
        self._backends = backends
        self._storage = storage
        self._jobs: dict[str, Job] = {}
        self._pending: list[str] = []
        self._lock = threading.Lock()
        self._queue: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, req: JobRequest) -> Job:
        if req.backend not in self._backends:
            raise ValidationError(f"Unknown backend '{req.backend}'")
        job = Job(id=uuid.uuid4().hex[:12], backend=req.backend, request=req)
        with self._lock:
            self._jobs[job.id] = job
            self._pending.append(job.id)
        self._queue.put(job.id)
        return job

    def snapshot(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            position = self._pending.index(job_id) + 1 if job_id in self._pending else None
            return {
                "id": job.id,
                "backend": job.backend,
                "status": job.status,
                "phase": job.phase,
                "step": job.step,
                "total_steps": job.total_steps,
                "queue_position": position,
                "result_id": job.result_id,
                "result": job.result,
                "rewritten_prompt": job.rewritten_prompt,
                "warning": job.warning,
                "error": job.error,
            }

    def wait_idle(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks:
            if time.monotonic() > deadline:
                raise TimeoutError("Job queue did not become idle")
            time.sleep(0.01)

    def _worker(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._run(job_id)
            finally:
                self._queue.task_done()

    def _run(self, job_id: str) -> None:
        with self._lock:
            self._pending.remove(job_id)
            job = self._jobs[job_id]
            job.status = "running"

        def on_progress(phase, step, total):
            job.phase, job.step, job.total_steps = phase, step, total

        started = time.monotonic()
        try:
            result = self._backends[job.backend].run(job.request, on_progress)
            meta = self._storage.save(job.request, result, time.monotonic() - started)
            job.result, job.result_id = meta, meta["id"]
            job.rewritten_prompt, job.warning = result.rewritten_prompt, result.warning
            job.status = "done"
        except BackendError as e:
            job.error, job.status = str(e), "error"
        except Exception as e:  # noqa: BLE001 - surface anything to the UI
            job.error, job.status = f"{type(e).__name__}: {e}", "error"
        finally:
            job.request = None  # free input images
