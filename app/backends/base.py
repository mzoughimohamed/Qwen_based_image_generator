from __future__ import annotations

from typing import Callable, Protocol

from app.schemas import JobRequest, JobResult

ProgressFn = Callable[[str, "int | None", "int | None"], None]


class BackendError(Exception):
    """A failure with a message that is safe to show to the user as-is."""


class Backend(Protocol):
    name: str

    def status(self) -> dict: ...

    def run(self, req: JobRequest, on_progress: ProgressFn) -> JobResult: ...
