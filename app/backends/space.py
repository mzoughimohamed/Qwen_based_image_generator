from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from PIL import Image

from app import config
from app.backends.base import BackendError, ProgressFn
from app.schemas import JobRequest, JobResult

SPACE_API_NAME = "/generate_with_enhance"
_UNUSED_SIZE = (2688, 1536)  # sent when custom_size is False; the Space ignores it
# The Space's "Log directory" textbox defaults to this value (confirmed via
# view_api()); we pass it through unchanged rather than an empty string so
# behavior matches the official UI's default.
_LOG_DIR = "./generation_logs_paper_case"
# The Space's "Seed" slider caps at 2^31 - 1 (confirmed via view_api()), lower
# than config.MAX_SEED (2^32 - 1) shared with the local backend.
SPACE_MAX_SEED = 2**31 - 1


def encode_images(paths: list[str]):
    if not paths:
        return None
    from gradio_client import handle_file

    return [{"image": handle_file(p), "caption": None} for p in paths]


def build_space_args(req: JobRequest, prompt: str, images_payload) -> list:
    width, height = req.size or _UNUSED_SIZE
    return [
        images_payload,                              # input_images
        prompt,                                      # prompt (original_prompt)
        req.enhance,                                 # enable_extend
        req.size is not None,                        # custom_size
        _LOG_DIR,                                     # log_dir
        req.seed if req.seed is not None else 0,     # seed
        req.seed is None,                            # randomize_seed
        height,                                      # height
        width,                                       # width
        req.negative_prompt,                         # negative_prompt
    ]


def friendly_space_error(exc: Exception) -> str:
    msg = str(exc) or type(exc).__name__
    low = msg.lower()
    if "quota" in low:
        return "HF Space quota exceeded — set HF_TOKEN or use the Local backend."
    if any(s in low for s in ("sleeping", "paused", "could not fetch config", "runtime_error")):
        return "HF Space is unavailable (sleeping or changed). Try again later or use Local."
    return f"HF Space error: {msg}"


def _result_path(value) -> str:
    if isinstance(value, (list, tuple)) and value:
        return _result_path(value[0])
    if isinstance(value, dict):
        return _result_path(value.get("path") or value.get("image") or value.get("url"))
    return str(value)


def _default_client():
    from gradio_client import Client

    return Client(config.SPACE_ID, token=os.environ.get("HF_TOKEN") or None)


class SpaceBackend:
    name = "space"

    def __init__(self, client_factory=None, poll_interval: float = 1.0, timeout_s: float = None):
        self._client_factory = client_factory or _default_client
        self._client = None
        self._poll_interval = poll_interval
        self._timeout_s = config.SPACE_TIMEOUT_S if timeout_s is None else timeout_s

    def status(self) -> dict:
        return {"state": "ready", "detail": f"Remote: huggingface.co/spaces/{config.SPACE_ID}"}

    def run(self, req: JobRequest, on_progress: ProgressFn) -> JobResult:
        if req.seed is not None and req.seed > SPACE_MAX_SEED:
            raise BackendError(
                f"HF Space accepts seeds up to {SPACE_MAX_SEED}; use a smaller seed or the Local backend."
            )
        prompt = f"{req.prompt} {config.RGBA_PHRASE}" if req.transparent else req.prompt
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, img in enumerate(req.images):
                path = Path(tmp) / f"input-{i}.png"
                img.save(path)
                paths.append(str(path))
            try:
                if self._client is None:
                    self._client = self._client_factory()
                job = self._client.submit(
                    *build_space_args(req, prompt, encode_images(paths)), api_name=SPACE_API_NAME
                )
                deadline = time.monotonic() + self._timeout_s
                while not job.done():
                    if time.monotonic() > deadline:
                        try:
                            job.cancel()
                        except Exception:  # noqa: BLE001 - cancellation is best-effort
                            pass
                        raise BackendError(
                            f"HF Space did not respond within {self._timeout_s / 60:.0f} "
                            "minutes; try again later or use Local."
                        )
                    rank = getattr(job.status(), "rank", None)
                    on_progress("remote-queue", rank + 1 if isinstance(rank, int) else None, None)
                    time.sleep(self._poll_interval)
                image_out, seed, rewritten = job.result()
                with Image.open(_result_path(image_out)) as im:
                    im.load()
                    image = im.copy()
            except BackendError:
                raise
            except Exception as e:  # noqa: BLE001 - every remote failure becomes a user message
                self._client = None
                raise BackendError(friendly_space_error(e)) from e
        return JobResult(
            image=image, seed=int(seed), rewritten_prompt=rewritten if (req.enhance and rewritten) else None
        )
