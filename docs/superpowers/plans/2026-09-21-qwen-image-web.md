# Qwen-Image Web Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local single-user website that generates and edits images with Qwen-Image-2.1, running either on the local GPU or through the official HF Space.

**Architecture:** A FastAPI app (created with `create_app()` and run with `uvicorn --factory`) exposes a single job endpoint. A single-worker FIFO queue dispatches each job to a `Backend`: `LocalBackend` (Diffusers `QwenImage21Pipeline` plus an optional PE prompt enhancer) or `SpaceBackend` (`gradio_client`). Results are stored as PNG + JSON in `outputs/`. The frontend is static HTML/CSS/JS served by the same app.

**Tech Stack:**
- Python 3.12 in a uv venv
- torch (CUDA 12.8 wheels)
- diffusers (git)
- transformers>=5.17
- accelerate
- gradio_client
- FastAPI, uvicorn
- Pillow
- pytest, httpx

**Spec:** `docs/superpowers/specs/2026-09-21-qwen-image-web-design.md`

## Global Constraints

- The server binds to `127.0.0.1` only. There is no authentication.
- Python 3.12 venv at `.venv` (the system Python is 3.14 and must not be used for the app).
- Models:
  - `Qwen/Qwen-Image-2.1`
  - enhancers `Qwen/Qwen-Image-2.1-PE-T2I` and `Qwen/Qwen-Image-2.1-PE-I2I`
  - Space `Qwen/Qwen-Image-2.1`
- RGBA phrase, verbatim: `This is an RGBA image with transparency. The image has alpha channel and the background is transparent.`
- Limits:
  - 0–10 input images
  - 20 MB per upload
  - steps 1–100, default 40
  - seed 0–4294967295
  - custom size 256–2688, rounded to a multiple of 16
- Size presets:
  - 2K: 2688×1536 16:9, 1536×2688 9:16, 2048×2048 1:1, 2368×1728 4:3, 1728×2368 3:4
  - 1K: 1344×768 16:9, 768×1344 9:16, 1024×1024 1:1, 1184×864 4:3, 864×1184 3:4
- Default text-to-image size: 2048×2048.
- Default negative prompt: `" "` (a single space).
- Enhancer `max_new_tokens`: 4096 (configurable).
- All tests run without a GPU or network (fakes are injected).
- Commands run from the repo root in PowerShell. The Python used is `.\.venv\Scripts\python.exe`, written below as `py312`.

---

### Task 1: Project setup, config and request schemas

**Files:**
- Create: `.gitignore`, `requirements.txt`, `pytest.ini`, `app/__init__.py`, `app/config.py`, `app/schemas.py`, `tests/__init__.py`, `tests/test_schemas.py`

**Interfaces:**
- Produces:
  - `app.config` constants: `ROOT, OUTPUTS_DIR, STATIC_DIR, MODEL_ID, PE_T2I_ID, PE_I2I_ID, SPACE_ID, RGBA_PHRASE, MAX_IMAGES, MAX_UPLOAD_BYTES, MIN_STEPS, MAX_STEPS, DEFAULT_STEPS, MAX_SEED, CUSTOM_MIN, CUSTOM_MAX, DEFAULT_T2I_SIZE, ENHANCER_MAX_NEW_TOKENS, SIZE_PRESETS: dict[str, list[tuple[str, int, int]]]`
  - `app.schemas`:
    - `ValidationError(ValueError)` and `BACKENDS = ("local", "space")`
    - `JobRequest` dataclass (`backend, prompt, negative_prompt=" ", images=[], size: tuple[int,int]|None=None, steps=40, seed: int|None=None, transparent=False, enhance=False`, property `mode`)
    - `JobResult` dataclass (`image, seed, rewritten_prompt=None, warning=None`)
    - `parse_size(str|None) -> tuple|None`
    - `parse_ratio(str|None) -> float|None`
    - `nearest_preset(ratio: float, tier="2K") -> tuple[int,int]`
    - `resolve_local_size(size, images, ratio_hint=None) -> tuple[int,int]`
    - `size_label(w, h) -> str`
    - `normalize_image(img) -> Image`
    - `build_request(*, backend, prompt, negative_prompt=None, images=(), size=None, steps=None, seed=None, transparent=False, enhance=False) -> JobRequest`

- [ ] **Step 1: Create the venv and install dependencies**

```powershell
python -m pip install --user uv
python -m uv venv --python 3.12 .venv
python -m uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu128
```

Create `requirements.txt`:

```
diffusers @ git+https://github.com/huggingface/diffusers
transformers>=5.17
accelerate
huggingface_hub
gradio_client
fastapi
uvicorn[standard]
python-multipart
pillow
pytest
httpx
```

```powershell
python -m uv pip install --python .venv -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch, diffusers; print(torch.__version__, torch.cuda.is_available(), hasattr(diffusers, 'QwenImage21Pipeline'))"
```

Expected: the torch version prints, then `True True`.

Create `.gitignore`:

```
.venv/
outputs/
__pycache__/
.pytest_cache/
```

Create `pytest.ini`:

```ini
[pytest]
testpaths = tests
pythonpath = .
```

Create empty `app/__init__.py` and `tests/__init__.py`.

- [ ] **Step 2: Write the failing tests** in `tests/test_schemas.py`

```python
import pytest
from PIL import Image

from app import config
from app.schemas import (
    ValidationError,
    build_request,
    nearest_preset,
    normalize_image,
    parse_ratio,
    parse_size,
    resolve_local_size,
    size_label,
)


def img(w=8, h=8, mode="RGB"):
    return Image.new(mode, (w, h))


@pytest.mark.parametrize("value", [None, "", "auto", "AUTO", " auto "])
def test_parse_size_auto(value):
    assert parse_size(value) is None


def test_parse_size_rounds_to_multiple_of_16():
    assert parse_size("1000x1010") == (1008, 1008)
    assert parse_size("2048x2048") == (2048, 2048)


@pytest.mark.parametrize("value", ["100x2048", "4000x1024", "big", "12x", "x12"])
def test_parse_size_rejects_invalid(value):
    with pytest.raises(ValidationError):
        parse_size(value)


def test_parse_ratio():
    assert parse_ratio("16:9") == pytest.approx(16 / 9)
    assert parse_ratio("") is None
    assert parse_ratio(None) is None
    assert parse_ratio("abc") is None
    assert parse_ratio("1:0") is None


def test_nearest_preset():
    assert nearest_preset(1.7) == (2688, 1536)
    assert nearest_preset(0.75) == (1728, 2368)
    assert nearest_preset(1.0) == (2048, 2048)
    assert nearest_preset(1.0, tier="1K") == (1024, 1024)


def test_resolve_local_size_explicit_wins():
    assert resolve_local_size((1024, 1024), [img(300, 100)], "9:16") == (1024, 1024)


def test_resolve_local_size_ratio_hint_beats_image():
    assert resolve_local_size(None, [img(300, 100)], "9:16") == (1536, 2688)


def test_resolve_local_size_from_first_image():
    assert resolve_local_size(None, [img(400, 300), img(100, 300)]) == (2368, 1728)


def test_resolve_local_size_default():
    assert resolve_local_size(None, [], None) == config.DEFAULT_T2I_SIZE


def test_size_label():
    assert size_label(1344, 768) == "1K 16:9"
    assert size_label(2048, 2048) == "2K 1:1"
    assert size_label(1000, 1000) == "custom"


def test_build_request_defaults():
    req = build_request(backend="local", prompt="  a cat  ")
    assert req.prompt == "a cat"
    assert req.negative_prompt == " "
    assert req.steps == config.DEFAULT_STEPS
    assert req.seed is None
    assert req.size is None
    assert req.mode == "generate"


def test_build_request_edit_mode_and_size():
    req = build_request(backend="space", prompt="p", images=[img()], size="1024x1024",
                        negative_prompt="blurry", seed=0, steps=12, transparent=True, enhance=True)
    assert req.mode == "edit"
    assert req.size == (1024, 1024)
    assert req.negative_prompt == "blurry"
    assert (req.seed, req.steps, req.transparent, req.enhance) == (0, 12, True, True)


@pytest.mark.parametrize("kwargs", [
    {"backend": "cloud"},
    {"prompt": "   "},
    {"steps": 0},
    {"steps": 101},
    {"seed": -1},
    {"seed": 2**32},
    {"images": [Image.new("RGB", (8, 8)) for _ in range(11)]},
    {"size": "big"},
])
def test_build_request_rejects(kwargs):
    base = {"backend": "local", "prompt": "a cat"}
    base.update(kwargs)
    with pytest.raises(ValidationError):
        build_request(**base)


def test_normalize_image_modes():
    assert normalize_image(img(mode="RGBA")).mode == "RGBA"
    assert normalize_image(img(mode="LA")).mode == "RGBA"
    assert normalize_image(img(mode="L")).mode == "RGB"
    assert normalize_image(img(mode="CMYK")).mode == "RGB"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_schemas.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'app.schemas'`.

- [ ] **Step 4: Implement `app/config.py`**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"
STATIC_DIR = ROOT / "static"

MODEL_ID = "Qwen/Qwen-Image-2.1"
PE_T2I_ID = "Qwen/Qwen-Image-2.1-PE-T2I"
PE_I2I_ID = "Qwen/Qwen-Image-2.1-PE-I2I"
SPACE_ID = "Qwen/Qwen-Image-2.1"

RGBA_PHRASE = (
    "This is an RGBA image with transparency. "
    "The image has alpha channel and the background is transparent."
)

MAX_IMAGES = 10
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MIN_STEPS, MAX_STEPS, DEFAULT_STEPS = 1, 100, 40
MAX_SEED = 2**32 - 1
CUSTOM_MIN, CUSTOM_MAX = 256, 2688
DEFAULT_T2I_SIZE = (2048, 2048)
ENHANCER_MAX_NEW_TOKENS = 4096

# (aspect label, width, height), matching the official Space presets.
SIZE_PRESETS = {
    "2K": [
        ("16:9", 2688, 1536),
        ("9:16", 1536, 2688),
        ("1:1", 2048, 2048),
        ("4:3", 2368, 1728),
        ("3:4", 1728, 2368),
    ],
    "1K": [
        ("16:9", 1344, 768),
        ("9:16", 768, 1344),
        ("1:1", 1024, 1024),
        ("4:3", 1184, 864),
        ("3:4", 864, 1184),
    ],
}
```

- [ ] **Step 5: Implement `app/schemas.py`**

```python
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from PIL import Image, ImageOps

from app import config

BACKENDS = ("local", "space")

_SIZE_RE = re.compile(r"^(\d+)x(\d+)$")
_RATIO_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*$")


class ValidationError(ValueError):
    pass


@dataclass
class JobRequest:
    backend: str
    prompt: str
    negative_prompt: str = " "
    images: list[Image.Image] = field(default_factory=list)
    size: tuple[int, int] | None = None
    steps: int = config.DEFAULT_STEPS
    seed: int | None = None
    transparent: bool = False
    enhance: bool = False

    @property
    def mode(self) -> str:
        return "edit" if self.images else "generate"


@dataclass
class JobResult:
    image: Image.Image
    seed: int
    rewritten_prompt: str | None = None
    warning: str | None = None


def _round16(value: int) -> int:
    return int(value / 16 + 0.5) * 16


def parse_size(value: str | None) -> tuple[int, int] | None:
    if value is None or value.strip().lower() in ("", "auto"):
        return None
    match = _SIZE_RE.match(value.strip().lower())
    if not match:
        raise ValidationError(f"Invalid size '{value}' (use 'auto' or WIDTHxHEIGHT)")
    width, height = _round16(int(match.group(1))), _round16(int(match.group(2)))
    for dim in (width, height):
        if not config.CUSTOM_MIN <= dim <= config.CUSTOM_MAX:
            raise ValidationError(
                f"Size must be between {config.CUSTOM_MIN} and {config.CUSTOM_MAX} pixels per side"
            )
    return width, height


def parse_ratio(text: str | None) -> float | None:
    match = _RATIO_RE.match(text or "")
    if not match or float(match.group(2)) == 0:
        return None
    return float(match.group(1)) / float(match.group(2))


def nearest_preset(ratio: float, tier: str = "2K") -> tuple[int, int]:
    _, width, height = min(
        config.SIZE_PRESETS[tier], key=lambda p: abs(math.log(p[1] / p[2]) - math.log(ratio))
    )
    return width, height


def resolve_local_size(size, images, ratio_hint: str | None = None) -> tuple[int, int]:
    if size:
        return size
    ratio = parse_ratio(ratio_hint)
    if ratio:
        return nearest_preset(ratio)
    if images:
        width, height = images[0].size
        return nearest_preset(width / height)
    return config.DEFAULT_T2I_SIZE


def size_label(width: int, height: int) -> str:
    for tier, presets in config.SIZE_PRESETS.items():
        for label, w, h in presets:
            if (w, h) == (width, height):
                return f"{tier} {label}"
    return "custom"


def normalize_image(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    has_alpha = img.mode in ("RGBA", "LA", "PA") or (
        img.mode == "P" and "transparency" in img.info
    )
    return img.convert("RGBA" if has_alpha else "RGB")


def build_request(
    *,
    backend: str,
    prompt: str,
    negative_prompt: str | None = None,
    images=(),
    size: str | None = None,
    steps: int | None = None,
    seed: int | None = None,
    transparent: bool = False,
    enhance: bool = False,
) -> JobRequest:
    if backend not in BACKENDS:
        raise ValidationError(f"Unknown backend '{backend}'")
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValidationError("Prompt must not be empty")
    images = list(images)
    if len(images) > config.MAX_IMAGES:
        raise ValidationError(f"At most {config.MAX_IMAGES} input images are allowed")
    steps = config.DEFAULT_STEPS if steps is None else int(steps)
    if not config.MIN_STEPS <= steps <= config.MAX_STEPS:
        raise ValidationError(f"Steps must be between {config.MIN_STEPS} and {config.MAX_STEPS}")
    if seed is not None and not 0 <= seed <= config.MAX_SEED:
        raise ValidationError(f"Seed must be between 0 and {config.MAX_SEED}")
    return JobRequest(
        backend=backend,
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt and negative_prompt.strip() else " ",
        images=images,
        size=parse_size(size),
        steps=steps,
        seed=seed,
        transparent=bool(transparent),
        enhance=bool(enhance),
    )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_schemas.py -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```powershell
git add .gitignore requirements.txt pytest.ini app tests
git commit -m "feat: project setup, config and request schemas"
```

---

### Task 2: Result storage

**Files:**
- Create: `app/storage.py`, `tests/test_storage.py`

**Interfaces:**
- Consumes: `JobRequest`, `JobResult`, `size_label` from `app.schemas`.
- Produces: `Storage(root: Path, clock=datetime.now)`, with these members:
  - `.root`
  - `.new_id() -> str`
  - `.save(req, result, duration_s) -> dict`
  - `.get(rid) -> dict | None`
  - `.list_history() -> list[dict]`
  - `.load_image(rid) -> Image | None`
  - `.delete(rid) -> bool`

  Metadata keys:
  - `id, mode, backend`
  - `prompt, rewritten_prompt, negative_prompt`
  - `width, height, size_label, steps, seed`
  - `transparent, enhance, warning`
  - `file, input_files, duration_s, created_at`

- [ ] **Step 1: Write the failing tests** in `tests/test_storage.py`

```python
from datetime import datetime, timedelta

from PIL import Image

from app.schemas import JobRequest, JobResult
from app.storage import Storage


class FakeClock:
    def __init__(self):
        self.now = datetime(2026, 9, 21, 12, 0, 0)

    def __call__(self):
        self.now += timedelta(seconds=1)
        return self.now


def make(tmp_path):
    return Storage(tmp_path / "out", clock=FakeClock())


def result(mode="RGB", size=(64, 32)):
    color = (0, 0, 0, 0) if mode == "RGBA" else (255, 0, 0)
    return JobResult(image=Image.new(mode, size, color), seed=7)


def test_save_writes_png_and_metadata(tmp_path):
    s = make(tmp_path)
    req = JobRequest(backend="local", prompt="a cat", seed=7, steps=30)
    meta = s.save(req, result(), duration_s=1.234)
    assert (s.root / meta["file"]).exists()
    assert meta["mode"] == "generate"
    assert meta["backend"] == "local"
    assert (meta["width"], meta["height"]) == (64, 32)
    assert meta["steps"] == 30
    assert meta["seed"] == 7
    assert meta["duration_s"] == 1.2
    assert meta["input_files"] == []
    assert meta["size_label"] == "custom"
    assert s.get(meta["id"]) == meta


def test_edit_inputs_are_saved(tmp_path):
    s = make(tmp_path)
    req = JobRequest(backend="local", prompt="p", images=[Image.new("RGB", (4, 4))] * 2)
    meta = s.save(req, result(), duration_s=0)
    assert meta["mode"] == "edit"
    assert len(meta["input_files"]) == 2
    for name in meta["input_files"]:
        assert (s.root / name).exists()


def test_rgba_is_preserved(tmp_path):
    s = make(tmp_path)
    meta = s.save(JobRequest(backend="local", prompt="p"), result("RGBA"), duration_s=0)
    with Image.open(s.root / meta["file"]) as im:
        assert im.mode == "RGBA"


def test_space_results_have_no_steps(tmp_path):
    s = make(tmp_path)
    meta = s.save(JobRequest(backend="space", prompt="p"), result(), duration_s=0)
    assert meta["steps"] is None


def test_list_history_newest_first(tmp_path):
    s = make(tmp_path)
    ids = [s.save(JobRequest(backend="local", prompt=str(i)), result(), 0)["id"] for i in range(3)]
    assert [m["id"] for m in s.list_history()] == list(reversed(ids))


def test_delete_removes_all_files(tmp_path):
    s = make(tmp_path)
    req = JobRequest(backend="local", prompt="p", images=[Image.new("RGB", (4, 4))])
    meta = s.save(req, result(), 0)
    assert s.delete(meta["id"]) is True
    assert list(s.root.iterdir()) == []
    assert s.delete(meta["id"]) is False


def test_bad_ids_are_rejected(tmp_path):
    s = make(tmp_path)
    assert s.get("../secret") is None
    assert s.delete("..") is False
    assert s.load_image("nope") is None


def test_load_image(tmp_path):
    s = make(tmp_path)
    meta = s.save(JobRequest(backend="local", prompt="p"), result(size=(10, 20)), 0)
    assert s.load_image(meta["id"]).size == (10, 20)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_storage.py -q`
Expected: `ModuleNotFoundError: No module named 'app.storage'`.

- [ ] **Step 3: Implement `app/storage.py`**

```python
from __future__ import annotations

import json
import re
import secrets
from datetime import datetime
from pathlib import Path

from PIL import Image

from app.schemas import JobRequest, JobResult, size_label

_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")


def _savable(img: Image.Image) -> Image.Image:
    if img.mode in ("RGB", "RGBA"):
        return img
    has_alpha = "A" in img.mode or "transparency" in img.info
    return img.convert("RGBA" if has_alpha else "RGB")


class Storage:
    def __init__(self, root: Path, clock=datetime.now):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._clock = clock

    def new_id(self, now: datetime | None = None) -> str:
        now = now or self._clock()
        return f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"

    def save(self, req: JobRequest, result: JobResult, duration_s: float) -> dict:
        now = self._clock()
        rid = self.new_id(now)
        while (self.root / f"{rid}.json").exists():
            rid = self.new_id(now)

        input_files = []
        for i, img in enumerate(req.images):
            name = f"{rid}.input-{i}.png"
            _savable(img).save(self.root / name)
            input_files.append(name)

        image = _savable(result.image)
        image.save(self.root / f"{rid}.png")
        width, height = image.size
        meta = {
            "id": rid,
            "mode": req.mode,
            "backend": req.backend,
            "prompt": req.prompt,
            "rewritten_prompt": result.rewritten_prompt,
            "negative_prompt": req.negative_prompt,
            "width": width,
            "height": height,
            "size_label": size_label(width, height),
            "steps": req.steps if req.backend == "local" else None,
            "seed": result.seed,
            "transparent": req.transparent,
            "enhance": req.enhance,
            "warning": result.warning,
            "file": f"{rid}.png",
            "input_files": input_files,
            "duration_s": round(duration_s, 1),
            "created_at": now.isoformat(timespec="microseconds"),
        }
        (self.root / f"{rid}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    def get(self, rid: str) -> dict | None:
        if not _ID_RE.match(rid or ""):
            return None
        path = self.root / f"{rid}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_history(self) -> list[dict]:
        metas = [json.loads(p.read_text(encoding="utf-8")) for p in self.root.glob("*.json")]
        return sorted(metas, key=lambda m: (m["created_at"], m["id"]), reverse=True)

    def load_image(self, rid: str) -> Image.Image | None:
        meta = self.get(rid)
        if meta is None:
            return None
        with Image.open(self.root / meta["file"]) as im:
            im.load()
            return im.copy()

    def delete(self, rid: str) -> bool:
        meta = self.get(rid)
        if meta is None:
            return False
        for name in [meta["file"], f"{rid}.json", *meta["input_files"]]:
            (self.root / name).unlink(missing_ok=True)
        return True
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_storage.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add app/storage.py tests/test_storage.py
git commit -m "feat: PNG + JSON result storage"
```

---

### Task 3: Backend protocol, test fake and job queue

**Files:**
- Create: `app/backends/__init__.py` (empty), `app/backends/base.py`, `app/jobs.py`, `tests/fakes.py`, `tests/test_jobs.py`

**Interfaces:**
- Consumes: `JobRequest`, `JobResult`, `ValidationError`, `Storage`.
- Produces:
  - `app.backends.base`: `BackendError(Exception)`, `ProgressFn = Callable[[str, int | None, int | None], None]`, `Backend` Protocol (`name`, `status() -> dict`, `run(req, on_progress) -> JobResult`).
  - `app.jobs`: `JobQueue(backends: dict[str, Backend], storage: Storage)`, with these methods:
    - `.submit(req) -> Job` (raises `ValidationError` for an unknown backend)
    - `.snapshot(job_id) -> dict | None`
    - `.wait_idle(timeout=10.0)`

  `snapshot` returns a dict with these keys:
  - `id, backend, status, phase, step, total_steps, queue_position`
  - `result_id, result` (the metadata dict)
  - `rewritten_prompt, warning, error`

  - `tests.fakes.FakeBackend(name="local", state="ready", error=None, rewritten=None, warning=None, mode="RGB", gate=None)`, with `.calls: list[JobRequest]` and `.started: threading.Event`.

- [ ] **Step 1: Write the fake and the failing tests**

`tests/fakes.py`:

```python
import threading

from PIL import Image

from app.schemas import JobResult


class FakeBackend:
    def __init__(self, name="local", state="ready", error=None, rewritten=None,
                 warning=None, mode="RGB", gate=None):
        self.name = name
        self.state = state
        self.error = error
        self.rewritten = rewritten
        self.warning = warning
        self.mode = mode
        self.gate = gate
        self.calls = []
        self.started = threading.Event()

    def status(self):
        return {"state": self.state, "detail": f"fake {self.name}"}

    def run(self, req, on_progress):
        self.calls.append(req)
        self.started.set()
        if self.gate is not None:
            self.gate.wait(5)
        if self.error:
            raise self.error
        for step in (1, 2):
            on_progress("generating", step, 2)
        width, height = req.size or (64, 64)
        color = (255, 0, 0, 0) if self.mode == "RGBA" else (255, 0, 0)
        return JobResult(
            image=Image.new(self.mode, (width, height), color),
            seed=req.seed if req.seed is not None else 123,
            rewritten_prompt=self.rewritten,
            warning=self.warning,
        )
```

`tests/test_jobs.py`:

```python
import threading

import pytest

from app.backends.base import BackendError
from app.jobs import JobQueue
from app.schemas import ValidationError, build_request
from app.storage import Storage
from tests.fakes import FakeBackend


def make_queue(tmp_path, **backends):
    return JobQueue(backends, Storage(tmp_path))


def test_job_completes_and_saves(tmp_path):
    q = make_queue(tmp_path, local=FakeBackend("local"))
    job = q.submit(build_request(backend="local", prompt="a cat"))
    q.wait_idle()
    snap = q.snapshot(job.id)
    assert snap["status"] == "done"
    assert snap["result_id"] == snap["result"]["id"]
    assert snap["result"]["prompt"] == "a cat"
    assert (snap["phase"], snap["step"], snap["total_steps"]) == ("generating", 2, 2)
    assert snap["queue_position"] is None


def test_fifo_order_and_queue_positions(tmp_path):
    gate = threading.Event()
    fake = FakeBackend("local", gate=gate)
    q = make_queue(tmp_path, local=fake)
    jobs = [q.submit(build_request(backend="local", prompt=str(i))) for i in range(3)]
    assert fake.started.wait(5)
    assert q.snapshot(jobs[0].id)["status"] == "running"
    assert q.snapshot(jobs[1].id)["queue_position"] == 1
    assert q.snapshot(jobs[2].id)["queue_position"] == 2
    gate.set()
    q.wait_idle()
    assert [r.prompt for r in fake.calls] == ["0", "1", "2"]


def test_backend_error_message(tmp_path):
    q = make_queue(tmp_path, local=FakeBackend("local", error=BackendError("boom")))
    job = q.submit(build_request(backend="local", prompt="p"))
    q.wait_idle()
    snap = q.snapshot(job.id)
    assert (snap["status"], snap["error"]) == ("error", "boom")


def test_unexpected_exception_is_reported(tmp_path):
    q = make_queue(tmp_path, local=FakeBackend("local", error=RuntimeError("bad")))
    job = q.submit(build_request(backend="local", prompt="p"))
    q.wait_idle()
    assert q.snapshot(job.id)["error"] == "RuntimeError: bad"


def test_rewritten_prompt_and_warning_propagate(tmp_path):
    fake = FakeBackend("local", rewritten="rich", warning="careful")
    q = make_queue(tmp_path, local=fake)
    job = q.submit(build_request(backend="local", prompt="p", enhance=True))
    q.wait_idle()
    snap = q.snapshot(job.id)
    assert (snap["rewritten_prompt"], snap["warning"]) == ("rich", "careful")
    assert snap["result"]["rewritten_prompt"] == "rich"


def test_unknown_backend_rejected(tmp_path):
    q = make_queue(tmp_path, local=FakeBackend("local"))
    with pytest.raises(ValidationError):
        q.submit(build_request(backend="space", prompt="p"))


def test_unknown_job_snapshot_is_none(tmp_path):
    assert make_queue(tmp_path, local=FakeBackend()).snapshot("missing") is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_jobs.py -q`
Expected: `ModuleNotFoundError: No module named 'app.backends'`.

- [ ] **Step 3: Implement `app/backends/base.py`**

```python
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
```

- [ ] **Step 4: Implement `app/jobs.py`**

```python
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_jobs.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```powershell
git add app/backends tests/fakes.py tests/test_jobs.py app/jobs.py
git commit -m "feat: backend protocol and single-worker job queue"
```

---

### Task 4: HTTP API

**Files:**
- Create: `app/main.py`, `static/index.html` (minimal shell; Task 8 replaces it), `tests/test_api.py`

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces:
  - `app.main.create_app(backends: dict | None = None, storage: Storage | None = None) -> FastAPI`, with `app.state.jobs: JobQueue` and `app.state.storage: Storage`.
  - `app.main.default_backends() -> dict`, which imports `LocalBackend` (Task 6) and `SpaceBackend` (Task 7) lazily inside the function.
  - Routes as in the spec.

- [ ] **Step 1: Write the failing tests** in `tests/test_api.py`

```python
import io
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import config
from app.backends.base import BackendError
from app.main import create_app
from app.storage import Storage
from tests.fakes import FakeBackend


@pytest.fixture
def ctx(tmp_path):
    backends = {"local": FakeBackend("local"), "space": FakeBackend("space")}
    app = create_app(backends=backends, storage=Storage(tmp_path))
    return SimpleNamespace(client=TestClient(app), app=app, backends=backends)


def png(size=(32, 32), mode="RGB"):
    buf = io.BytesIO()
    Image.new(mode, size).save(buf, "PNG")
    return buf.getvalue()


def run_job(ctx, data, files=None):
    r = ctx.client.post("/api/jobs", data=data, files=files)
    assert r.status_code == 200, r.text
    ctx.app.state.jobs.wait_idle()
    return ctx.client.get(f"/api/jobs/{r.json()['job_id']}").json()


def test_health_lists_backends(ctx):
    body = ctx.client.get("/api/health").json()
    assert body["backends"]["local"]["state"] == "ready"
    assert body["backends"]["space"]["state"] == "ready"


def test_options(ctx):
    body = ctx.client.get("/api/options").json()
    assert body["sizes"]["2K"][0] == {"label": "16:9", "width": 2688, "height": 1536}
    assert body["backends"] == ["local", "space"]
    assert body["limits"]["max_images"] == 10


def test_generate_job(ctx):
    job = run_job(ctx, {"backend": "local", "prompt": "a cat", "size": "1024x1024",
                        "seed": "5", "steps": "12", "transparent": "false", "enhance": "false"})
    assert job["status"] == "done"
    meta = job["result"]
    assert (meta["width"], meta["seed"], meta["steps"]) == (1024, 5, 12)
    assert ctx.client.get(f"/outputs/{meta['file']}").status_code == 200
    assert [m["id"] for m in ctx.client.get("/api/history").json()] == [meta["id"]]


def test_edit_with_upload_and_from_result(ctx):
    first = run_job(ctx, {"backend": "local", "prompt": "base"})["result"]
    job = run_job(ctx, {"backend": "space", "prompt": "combine", "from_result": [first["id"]]},
                  files=[("images", ("a.png", png(), "image/png"))])
    assert job["status"] == "done"
    req = ctx.backends["space"].calls[-1]
    assert req.mode == "edit"
    assert len(req.images) == 2
    assert len(job["result"]["input_files"]) == 2


@pytest.mark.parametrize("data", [
    {"backend": "local", "prompt": "   "},
    {"backend": "cloud", "prompt": "p"},
    {"backend": "local", "prompt": "p", "steps": "0"},
    {"backend": "local", "prompt": "p", "size": "huge"},
    {"backend": "local", "prompt": "p", "from_result": ["20260101-000000-abcd"]},
])
def test_rejects_invalid(ctx, data):
    assert ctx.client.post("/api/jobs", data=data).status_code == 400


def test_rejects_too_many_images(ctx):
    files = [("images", (f"{i}.png", png(), "image/png")) for i in range(11)]
    r = ctx.client.post("/api/jobs", data={"backend": "local", "prompt": "p"}, files=files)
    assert r.status_code == 400


def test_rejects_non_image(ctx):
    files = [("images", ("a.txt", b"hello", "text/plain"))]
    r = ctx.client.post("/api/jobs", data={"backend": "local", "prompt": "p"}, files=files)
    assert r.status_code == 400
    assert "not a valid image" in r.json()["detail"]


def test_rejects_oversize(ctx, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 100)
    files = [("images", ("a.png", png((64, 64)), "image/png"))]
    r = ctx.client.post("/api/jobs", data={"backend": "local", "prompt": "p"}, files=files)
    assert r.status_code == 400


def test_backend_not_ready_is_503(ctx):
    ctx.backends["local"].state = "loading"
    r = ctx.client.post("/api/jobs", data={"backend": "local", "prompt": "p"})
    assert r.status_code == 503


def test_job_error_is_reported(ctx):
    ctx.backends["local"].error = BackendError("Out of GPU memory")
    job = run_job(ctx, {"backend": "local", "prompt": "p"})
    assert (job["status"], job["error"]) == ("error", "Out of GPU memory")


def test_delete_history(ctx):
    rid = run_job(ctx, {"backend": "local", "prompt": "p"})["result"]["id"]
    assert ctx.client.delete(f"/api/history/{rid}").status_code == 204
    assert ctx.client.delete(f"/api/history/{rid}").status_code == 404
    assert ctx.client.get("/api/history").json() == []


def test_unknown_job_is_404(ctx):
    assert ctx.client.get("/api/jobs/nope").status_code == 404


def test_index_is_served(ctx):
    r = ctx.client.get("/")
    assert r.status_code == 200
    assert "Qwen-Image" in r.text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_api.py -q`
Expected: `ModuleNotFoundError: No module named 'app.main'`.

- [ ] **Step 3: Create a minimal `static/index.html`**

```html
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Qwen-Image Studio</title></head>
<body><h1>Qwen-Image Studio</h1></body></html>
```

- [ ] **Step 4: Implement `app/main.py`**

```python
from __future__ import annotations

import io

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from app import config
from app.jobs import JobQueue
from app.schemas import ValidationError, build_request, normalize_image
from app.storage import Storage


def default_backends() -> dict:
    from app.backends.local import LocalBackend
    from app.backends.space import SpaceBackend

    local = LocalBackend()
    local.start()
    return {"local": local, "space": SpaceBackend()}


async def _read_upload(upload: UploadFile) -> Image.Image:
    data = await upload.read(config.MAX_UPLOAD_BYTES + 1)
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise ValidationError(f"'{upload.filename}' is larger than 20 MB")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise ValidationError(f"'{upload.filename}' is not a valid image") from None
    return normalize_image(img)


def create_app(backends: dict | None = None, storage: Storage | None = None) -> FastAPI:
    backends = backends if backends is not None else default_backends()
    storage = storage or Storage(config.OUTPUTS_DIR)
    jobs = JobQueue(backends, storage)

    app = FastAPI(title="Qwen-Image Studio")
    app.state.jobs = jobs
    app.state.storage = storage

    @app.get("/api/health")
    def health():
        return {"backends": {name: b.status() for name, b in backends.items()}}

    @app.get("/api/options")
    def options():
        return {
            "sizes": {
                tier: [{"label": label, "width": w, "height": h} for label, w, h in presets]
                for tier, presets in config.SIZE_PRESETS.items()
            },
            "backends": list(backends),
            "limits": {
                "max_images": config.MAX_IMAGES,
                "max_upload_bytes": config.MAX_UPLOAD_BYTES,
                "min_steps": config.MIN_STEPS,
                "max_steps": config.MAX_STEPS,
                "default_steps": config.DEFAULT_STEPS,
                "max_seed": config.MAX_SEED,
                "custom_min": config.CUSTOM_MIN,
                "custom_max": config.CUSTOM_MAX,
            },
        }

    @app.post("/api/jobs")
    async def create_job(
        backend: str = Form(...),
        prompt: str = Form(...),
        negative_prompt: str | None = Form(None),
        size: str | None = Form(None),
        steps: int | None = Form(None),
        seed: int | None = Form(None),
        transparent: bool = Form(False),
        enhance: bool = Form(False),
        images: list[UploadFile] = File(default=[]),
        from_result: list[str] = Form(default=[]),
    ):
        try:
            if backend not in backends:
                raise ValidationError(f"Unknown backend '{backend}'")
            if len(images) + len(from_result) > config.MAX_IMAGES:
                raise ValidationError(f"At most {config.MAX_IMAGES} input images are allowed")
            pil_images = [await _read_upload(f) for f in images]
            for rid in from_result:
                img = storage.load_image(rid)
                if img is None:
                    raise ValidationError(f"Unknown result id '{rid}'")
                pil_images.append(normalize_image(img))
            req = build_request(
                backend=backend, prompt=prompt, negative_prompt=negative_prompt,
                images=pil_images, size=size, steps=steps, seed=seed,
                transparent=transparent, enhance=enhance,
            )
        except ValidationError as e:
            raise HTTPException(400, str(e)) from None

        status = backends[backend].status()
        if status["state"] != "ready":
            raise HTTPException(503, f"Backend '{backend}' is {status['state']}: {status['detail']}")
        return {"job_id": jobs.submit(req).id}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        snap = jobs.snapshot(job_id)
        if snap is None:
            raise HTTPException(404, "Unknown job")
        return snap

    @app.get("/api/history")
    def history():
        return storage.list_history()

    @app.delete("/api/history/{rid}")
    def delete_history(rid: str):
        if not storage.delete(rid):
            raise HTTPException(404, "Unknown result")
        return Response(status_code=204)

    app.mount("/outputs", StaticFiles(directory=storage.root), name="outputs")
    app.mount("/", StaticFiles(directory=config.STATIC_DIR, html=True), name="static")
    return app
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest -q`
Expected: all tests (Tasks 1–4) pass.

- [ ] **Step 6: Commit**

```powershell
git add app/main.py static/index.html tests/test_api.py
git commit -m "feat: FastAPI job, history and health endpoints"
```

---

### Task 5: Prompt enhancer

**Files:**
- Create: `app/backends/enhancer.py`, `tests/test_enhancer.py`

**Interfaces:**
- Consumes: `config.PE_T2I_ID`, `config.PE_I2I_ID`, `config.ENHANCER_MAX_NEW_TOKENS`.
- Produces:
  - `free_cuda() -> None`
  - `parse_enhancer_output(text) -> tuple[str, str | None]`, which raises `ValueError`
  - `LoadedModel(model, processor, system_prompt: str | None, kind: "t2i" | "i2i")`
  - `load_pe_model(model_id) -> LoadedModel`
  - `run_pe_model(lm, prompt, images, max_new_tokens, device) -> str`
  - `PromptEnhancer(load_fn=load_pe_model, run_fn=run_pe_model, device="cuda", max_new_tokens=config.ENHANCER_MAX_NEW_TOKENS)`, with `.enhance(prompt, images) -> tuple[str, str | None]` (raises `ValueError` when the output can't be parsed), `.unload()` and `.resident_id`

- [ ] **Step 1: Write the failing tests** in `tests/test_enhancer.py`

```python
import pytest
from PIL import Image

from app import config
from app.backends.enhancer import LoadedModel, PromptEnhancer, parse_enhancer_output


def test_parse_after_think_block():
    text = ('<think>draft {"rewritten_prompt": "draft"}</think>\n'
            '{"rewritten_prompt": "A red fox", "wh_ratio": "16:9"}<|im_end|>')
    assert parse_enhancer_output(text) == ("A red fox", "16:9")


def test_parse_without_think_and_empty_ratio():
    assert parse_enhancer_output('{"rewritten_prompt": "x", "wh_ratio": ""}') == ("x", None)


def test_parse_i2i_extra_keys():
    text = '</think>{"rewritten_prompt": "y", "wh_ratio": "", "ratio_follow": "<image1>"}'
    assert parse_enhancer_output(text) == ("y", None)


def test_parse_takes_last_object():
    text = '{"rewritten_prompt": "a"} and then {"rewritten_prompt": "b", "wh_ratio": "1:1"}'
    assert parse_enhancer_output(text) == ("b", "1:1")


@pytest.mark.parametrize("text", ["no json here", '{"rewritten_prompt": ""}', '{"other": 1}'])
def test_parse_malformed_raises(text):
    with pytest.raises(ValueError):
        parse_enhancer_output(text)


class FakeModel:
    def __init__(self):
        self.devices = []

    def to(self, device):
        self.devices.append(device)
        return self


GOOD = '{"rewritten_prompt": "rich", "wh_ratio": "9:16"}'


def make(outputs):
    loads, models = [], {}

    def load_fn(model_id):
        loads.append(model_id)
        models[model_id] = FakeModel()
        kind = "i2i" if model_id == config.PE_I2I_ID else "t2i"
        return LoadedModel(models[model_id], None, "sys", kind)

    def run_fn(lm, prompt, images, max_new_tokens, device):
        out = outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    return PromptEnhancer(load_fn=load_fn, run_fn=run_fn, device="gpu"), loads, models


def test_lazy_load_and_reuse():
    enh, loads, models = make([GOOD, GOOD])
    assert enh.resident_id is None
    assert enh.enhance("p", []) == ("rich", "9:16")
    assert enh.enhance("p", []) == ("rich", "9:16")
    assert loads == [config.PE_T2I_ID]
    assert models[config.PE_T2I_ID].devices == ["gpu", "cpu", "gpu", "cpu"]


def test_switching_mode_unloads_other():
    enh, loads, models = make([GOOD, GOOD])
    enh.enhance("p", [])
    enh.enhance("p", [Image.new("RGB", (4, 4))])
    assert loads == [config.PE_T2I_ID, config.PE_I2I_ID]
    assert enh.resident_id == config.PE_I2I_ID
    assert models[config.PE_T2I_ID].devices[-1] == "cpu"


def test_generation_error_unloads():
    enh, _, _ = make([RuntimeError("CUDA out of memory")])
    with pytest.raises(RuntimeError):
        enh.enhance("p", [])
    assert enh.resident_id is None


def test_unparseable_output_keeps_model():
    enh, _, models = make(["garbage"])
    with pytest.raises(ValueError):
        enh.enhance("p", [])
    assert enh.resident_id == config.PE_T2I_ID
    assert models[config.PE_T2I_ID].devices[-1] == "cpu"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_enhancer.py -q`
Expected: `ModuleNotFoundError: No module named 'app.backends.enhancer'`.

- [ ] **Step 3: Implement `app/backends/enhancer.py`**

The load and run code follows the model cards: `PE-T2I` uses `AutoModelForCausalLM` with a tokenizer, and `PE-I2I` uses `AutoModelForImageTextToText` with a processor and images in the chat content. Models load onto the CPU and move to the GPU only while they run.

```python
from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from app import config

_DECODER = json.JSONDecoder()


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_enhancer_output(text: str) -> tuple[str, str | None]:
    tail = text.rsplit("</think>", 1)[-1]
    found = None
    for i, ch in enumerate(tail):
        if ch != "{":
            continue
        try:
            obj, _ = _DECODER.raw_decode(tail, i)
        except json.JSONDecodeError:
            continue
        prompt = obj.get("rewritten_prompt") if isinstance(obj, dict) else None
        if isinstance(prompt, str) and prompt.strip():
            found = obj
    if found is None:
        raise ValueError("enhancer output had no rewritten_prompt JSON")
    ratio = found.get("wh_ratio")
    return found["rewritten_prompt"].strip(), ratio if isinstance(ratio, str) and ratio else None


@dataclass
class LoadedModel:
    model: Any
    processor: Any
    system_prompt: str | None
    kind: str  # "t2i" | "i2i"


def _system_prompt(model_id: str) -> str | None:
    import huggingface_hub

    try:
        path = huggingface_hub.hf_hub_download(model_id, "system_prompt.txt")
    except Exception:  # noqa: BLE001 - repo may not ship one
        return None
    return Path(path).read_text(encoding="utf-8").strip()


def load_pe_model(model_id: str) -> LoadedModel:
    if model_id == config.PE_I2I_ID:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16).eval()
        processor = AutoProcessor.from_pretrained(model_id)
        kind = "i2i"
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).eval()
        processor = AutoTokenizer.from_pretrained(model_id)
        kind = "t2i"
    return LoadedModel(model, processor, _system_prompt(model_id), kind)


def run_pe_model(lm: LoadedModel, prompt: str, images, max_new_tokens: int, device: str) -> str:
    if lm.kind == "i2i":
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        if lm.system_prompt:
            messages.insert(0, {"role": "system",
                                "content": [{"type": "text", "text": lm.system_prompt}]})
        inputs = lm.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", enable_thinking=True,
        ).to(device)
        tokenizer = lm.processor.tokenizer
    else:
        messages = [{"role": "user", "content": prompt}]
        if lm.system_prompt:
            messages.insert(0, {"role": "system", "content": lm.system_prompt})
        text = lm.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        inputs = lm.processor(text, return_tensors="pt").to(device)
        tokenizer = lm.processor
    with torch.no_grad():
        out = lm.model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=1.0, top_p=0.95, top_k=20,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)


class PromptEnhancer:
    def __init__(self, load_fn=load_pe_model, run_fn=run_pe_model, device: str = "cuda",
                 max_new_tokens: int = config.ENHANCER_MAX_NEW_TOKENS):
        self._load_fn = load_fn
        self._run_fn = run_fn
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._resident: LoadedModel | None = None
        self._resident_id: str | None = None

    @property
    def resident_id(self) -> str | None:
        return self._resident_id

    def enhance(self, prompt: str, images) -> tuple[str, str | None]:
        model_id = config.PE_I2I_ID if images else config.PE_T2I_ID
        lm = self._ensure(model_id)
        try:
            lm.model.to(self._device)
            text = self._run_fn(lm, prompt, images, self._max_new_tokens, self._device)
        except Exception:
            self.unload()
            raise
        lm.model.to("cpu")
        free_cuda()
        return parse_enhancer_output(text)

    def unload(self) -> None:
        if self._resident is not None:
            try:
                self._resident.model.to("cpu")
            except Exception:  # noqa: BLE001 - best effort while recovering
                pass
        self._resident = None
        self._resident_id = None
        free_cuda()

    def _ensure(self, model_id: str) -> LoadedModel:
        if self._resident_id != model_id:
            self.unload()
            self._resident = self._load_fn(model_id)
            self._resident_id = model_id
        return self._resident
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_enhancer.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add app/backends/enhancer.py tests/test_enhancer.py
git commit -m "feat: on-demand PE prompt enhancer"
```

---

### Task 6: Local GPU backend

**Files:**
- Create: `app/backends/local.py`, `tests/test_local_backend.py`

**Interfaces:**
- Consumes: `JobRequest`, `JobResult`, `resolve_local_size`, `BackendError`, `PromptEnhancer`, `free_cuda`, `config.RGBA_PHRASE`, `config.MODEL_ID`, `config.MAX_SEED`.
- Produces:
  - `load_pipeline(offload: bool)`
  - `LocalBackend(loader=load_pipeline, enhancer=None, device="cuda", enabled=True)`, with:
    - `name = "local"`
    - `.start()` (loads in a background thread)
    - `.load()` (synchronous)
    - `.status() -> {"state", "detail", "offload"}`
    - `.run(req, on_progress) -> JobResult`

- [ ] **Step 1: Write the failing tests** in `tests/test_local_backend.py`

```python
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from app import config
from app.backends.base import BackendError
from app.backends.local import LocalBackend
from app.schemas import build_request


class FakePipe:
    def __init__(self):
        self.calls = []
        self.devices = []
        self.offloaded = False

    def to(self, device):
        self.devices.append(device)
        return self

    def enable_model_cpu_offload(self):
        self.offloaded = True

    def __call__(self, prompt, width, height, num_inference_steps, generator,
                 callback_on_step_end=None, image=None, negative_prompt="<absent>"):
        self.calls.append(dict(prompt=prompt, width=width, height=height,
                               steps=num_inference_steps, image=image,
                               negative_prompt=negative_prompt))
        for i in range(num_inference_steps):
            callback_on_step_end(self, i, None, {})
        return SimpleNamespace(images=[Image.new("RGB", (width, height))])


class FakePipeNoNegative(FakePipe):
    def __call__(self, prompt, width, height, num_inference_steps, generator,
                 callback_on_step_end=None, image=None):
        return FakePipe.__call__(self, prompt, width, height, num_inference_steps,
                                 generator, callback_on_step_end, image)


class OOMPipe(FakePipe):
    def __call__(self, *args, **kwargs):
        raise torch.OutOfMemoryError("CUDA out of memory")


class FakeEnhancer:
    def __init__(self, result=("rich prompt", "9:16"), error=None):
        self.result, self.error, self.calls = result, error, []

    def enhance(self, prompt, images):
        self.calls.append((prompt, images))
        if self.error:
            raise self.error
        return self.result


def make_backend(pipe=None, enhancer=None):
    pipe = pipe or FakePipe()
    backend = LocalBackend(loader=lambda offload: pipe, enhancer=enhancer, device="cpu")
    backend.load()
    return backend, pipe


def req(**kwargs):
    kwargs.setdefault("prompt", "a cat")
    return build_request(backend="local", **kwargs)


def run(backend, request):
    events = []
    result = backend.run(request, lambda *e: events.append(e))
    return result, events


def test_load_ready():
    backend, _ = make_backend()
    assert backend.status() == {"state": "ready", "detail": "Loaded on GPU", "offload": False}


def test_oom_on_load_falls_back_to_offload():
    pipe = FakePipe()

    def loader(offload):
        if not offload:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return pipe

    backend = LocalBackend(loader=loader, device="cpu")
    backend.load()
    assert backend.status()["state"] == "ready"
    assert backend.status()["offload"] is True


def test_load_error_is_reported():
    def loader(offload):
        raise RuntimeError("no weights")

    backend = LocalBackend(loader=loader, device="cpu")
    backend.load()
    assert backend.status()["state"] == "error"
    assert "no weights" in backend.status()["detail"]


def test_disabled_backend():
    backend = LocalBackend(enabled=False)
    assert backend.status()["state"] == "disabled"


def test_generate_passes_parameters():
    backend, pipe = make_backend()
    result, events = run(backend, req(size="1024x1024", steps=3, seed=5, negative_prompt="blurry"))
    call = pipe.calls[-1]
    assert (call["prompt"], call["width"], call["height"], call["steps"]) == ("a cat", 1024, 1024, 3)
    assert call["negative_prompt"] == "blurry"
    assert call["image"] is None
    assert result.seed == 5
    assert events[-1] == ("generating", 3, 3)


def test_random_seed_when_missing():
    backend, _ = make_backend()
    result, _ = run(backend, req(steps=1))
    assert 0 <= result.seed <= config.MAX_SEED


def test_transparent_appends_rgba_phrase():
    backend, pipe = make_backend()
    run(backend, req(steps=1, transparent=True))
    assert pipe.calls[-1]["prompt"] == f"a cat {config.RGBA_PHRASE}"


def test_edit_passes_rgb_images_and_auto_size():
    backend, pipe = make_backend()
    run(backend, req(steps=1, images=[Image.new("RGBA", (400, 300))]))
    call = pipe.calls[-1]
    assert [im.mode for im in call["image"]] == ["RGB"]
    assert (call["width"], call["height"]) == (2368, 1728)


def test_negative_prompt_skipped_when_unsupported():
    backend, pipe = make_backend(pipe=FakePipeNoNegative())
    run(backend, req(steps=1, negative_prompt="blurry"))
    assert pipe.calls[-1]["negative_prompt"] == "<absent>"


def test_enhance_uses_rewritten_prompt_and_ratio():
    enhancer = FakeEnhancer()
    backend, pipe = make_backend(enhancer=enhancer)
    result, events = run(backend, req(steps=1, enhance=True))
    call = pipe.calls[-1]
    assert call["prompt"] == "rich prompt"
    assert (call["width"], call["height"]) == (1536, 2688)
    assert result.rewritten_prompt == "rich prompt"
    assert events[0] == ("enhancing", None, None)
    assert pipe.offloaded is True
    assert backend.status()["offload"] is True


def test_enhance_parse_failure_falls_back_to_original_prompt():
    backend, pipe = make_backend(enhancer=FakeEnhancer(error=ValueError("no json")))
    result, _ = run(backend, req(steps=1, enhance=True))
    assert pipe.calls[-1]["prompt"] == "a cat"
    assert result.rewritten_prompt is None
    assert "original prompt" in result.warning


def test_oom_during_generation_is_backend_error():
    backend, _ = make_backend(pipe=OOMPipe())
    with pytest.raises(BackendError, match="Out of GPU memory"):
        run(backend, req(steps=1))


def test_run_when_not_ready():
    backend = LocalBackend(loader=lambda offload: FakePipe(), device="cpu")
    with pytest.raises(BackendError):
        run(backend, req(steps=1))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_local_backend.py -q`
Expected: `ModuleNotFoundError: No module named 'app.backends.local'`.

- [ ] **Step 3: Implement `app/backends/local.py`**

```python
from __future__ import annotations

import inspect
import random
import threading

import torch

from app import config
from app.backends.base import BackendError, ProgressFn
from app.backends.enhancer import PromptEnhancer, free_cuda
from app.schemas import JobRequest, JobResult, resolve_local_size


def load_pipeline(offload: bool):
    from diffusers import QwenImage21Pipeline

    pipe = QwenImage21Pipeline.from_pretrained(config.MODEL_ID, torch_dtype=torch.bfloat16)
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


class LocalBackend:
    name = "local"

    def __init__(self, loader=load_pipeline, enhancer=None, device: str = "cuda",
                 enabled: bool = True):
        self._loader = loader
        self._enhancer = enhancer
        self._device = device
        self._pipe = None
        self._accepts_negative = False
        self._lock = threading.Lock()
        self.offload = False
        if enabled:
            self._state, self._detail = "loading", "Loading Qwen-Image-2.1…"
        else:
            self._state, self._detail = "disabled", "Disabled (QWEN_DISABLE_LOCAL=1)"

    def start(self) -> None:
        if self._state == "loading":
            threading.Thread(target=self.load, daemon=True).start()

    def load(self) -> None:
        try:
            try:
                pipe = self._loader(offload=False)
            except torch.OutOfMemoryError:
                free_cuda()
                pipe = self._loader(offload=True)
                self.offload = True
            self._pipe = pipe
            self._accepts_negative = "negative_prompt" in inspect.signature(pipe.__call__).parameters
            self._state = "ready"
            self._detail = "Loaded with CPU offload" if self.offload else "Loaded on GPU"
        except Exception as e:  # noqa: BLE001 - reported through status()
            self._state, self._detail = "error", f"{type(e).__name__}: {e}"

    def status(self) -> dict:
        return {"state": self._state, "detail": self._detail, "offload": self.offload}

    def _get_enhancer(self):
        if self._enhancer is None:
            self._enhancer = PromptEnhancer(device=self._device)
        return self._enhancer

    def _ensure_offload(self) -> None:
        if self.offload:
            return
        self._pipe.to("cpu")
        free_cuda()
        self._pipe.enable_model_cpu_offload()
        self.offload = True
        self._detail = "Loaded with CPU offload (prompt enhancer in use)"

    def run(self, req: JobRequest, on_progress: ProgressFn) -> JobResult:
        if self._state != "ready":
            raise BackendError(f"Local model is {self._state}: {self._detail}")
        with self._lock:
            rewritten = ratio = warning = None
            if req.enhance:
                on_progress("enhancing", None, None)
                self._ensure_offload()
                try:
                    rewritten, ratio = self._get_enhancer().enhance(req.prompt, req.images)
                except ValueError as e:
                    warning = f"Prompt enhancement failed ({e}); used your original prompt."
                except torch.OutOfMemoryError:
                    free_cuda()
                    raise BackendError("Out of GPU memory while enhancing the prompt.") from None

            prompt = rewritten or req.prompt
            if req.transparent:
                prompt = f"{prompt} {config.RGBA_PHRASE}"
            width, height = resolve_local_size(req.size, req.images, ratio)
            seed = req.seed if req.seed is not None else random.randint(0, config.MAX_SEED)

            def on_step(pipe, step, timestep, callback_kwargs):
                on_progress("generating", step + 1, req.steps)
                return callback_kwargs

            kwargs = dict(
                prompt=prompt, width=width, height=height,
                num_inference_steps=req.steps,
                generator=torch.Generator(self._device).manual_seed(seed),
                callback_on_step_end=on_step,
            )
            if req.images:
                kwargs["image"] = [im if req.transparent else im.convert("RGB") for im in req.images]
            if self._accepts_negative:
                kwargs["negative_prompt"] = req.negative_prompt

            on_progress("generating", 0, req.steps)
            try:
                with torch.inference_mode():
                    image = self._pipe(**kwargs).images[0]
            except torch.OutOfMemoryError:
                free_cuda()
                raise BackendError(
                    "Out of GPU memory — try a 1K size or fewer input images."
                ) from None
            return JobResult(image=image, seed=seed, rewritten_prompt=rewritten, warning=warning)
```

- [ ] **Step 4: Wire `QWEN_DISABLE_LOCAL` into `default_backends()`** in `app/main.py`:

```python
def default_backends() -> dict:
    import os

    from app.backends.local import LocalBackend
    from app.backends.space import SpaceBackend

    local = LocalBackend(enabled=os.environ.get("QWEN_DISABLE_LOCAL") != "1")
    local.start()
    return {"local": local, "space": SpaceBackend()}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```powershell
git add app/backends/local.py app/main.py tests/test_local_backend.py
git commit -m "feat: local Diffusers backend with enhancer and OOM handling"
```

---

### Task 7: HF Space backend

**Files:**
- Create: `app/backends/space.py`, `tests/test_space_backend.py`

**Interfaces:**
- Consumes: `JobRequest`, `JobResult`, `BackendError`, `config.SPACE_ID`, `config.RGBA_PHRASE`.
- Produces:
  - `SPACE_API_NAME: str`
  - `build_space_args(req, prompt, images_payload) -> list`
  - `encode_images(paths) -> list | None`
  - `friendly_space_error(exc) -> str`
  - `SpaceBackend(client_factory=None, poll_interval=1.0)`, with `name = "space"`, `.status()` and `.run(req, on_progress)`

- [ ] **Step 1: Check the Space's real API signature**

```powershell
.\.venv\Scripts\python.exe -c "from gradio_client import Client; Client('Qwen/Qwen-Image-2.1').view_api()"
```

Expected: an endpoint for `generate_with_enhance` whose 10 parameters are, in order: `input_images, prompt, enable_extend, custom_size, log_dir_input, seed, randomize_seed, height, width, negative_prompt_input`. It returns 3 values: `(image, seed, rewritten_prompt)`.

Record the values below, and if they differ, update `SPACE_API_NAME`, the argument order in `build_space_args`, `encode_images` and the tests to match. The code assumes:
- `api_name="/generate_with_enhance"`
- the gallery input is a list of `{"image": handle_file(path), "caption": None}`, or `None` when empty
- `log_dir_input` is `""`

- [ ] **Step 2: Write the failing tests** in `tests/test_space_backend.py`

```python
from types import SimpleNamespace

import pytest
from PIL import Image

from app import config
from app.backends.base import BackendError
from app.backends.space import SPACE_API_NAME, SpaceBackend, build_space_args, friendly_space_error
from app.schemas import build_request


def req(**kwargs):
    kwargs.setdefault("prompt", "p")
    return build_request(backend="space", **kwargs)


def test_args_auto_size_random_seed():
    assert build_space_args(req(), "p", None) == [None, "p", False, False, "", 0, True, 1536, 2688, " "]


def test_args_custom_size_fixed_seed_enhance():
    args = build_space_args(req(size="1024x768", seed=9, enhance=True, negative_prompt="ugly"), "p", None)
    assert args == [None, "p", True, True, "", 9, False, 768, 1024, "ugly"]


class FakeJob:
    def __init__(self, result, done_sequence=(False, True)):
        self._result = result
        self._done = list(done_sequence)

    def done(self):
        return self._done.pop(0) if len(self._done) > 1 else self._done[0]

    def status(self):
        return SimpleNamespace(rank=1)

    def result(self):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class FakeClient:
    def __init__(self, job=None, submit_error=None):
        self.job, self.submit_error, self.calls = job, submit_error, []

    def submit(self, *args, api_name):
        self.calls.append((args, api_name))
        if self.submit_error:
            raise self.submit_error
        return self.job


def output_png(tmp_path, mode="RGBA"):
    path = tmp_path / "out.png"
    Image.new(mode, (40, 20)).save(path)
    return str(path)


def make(client):
    return SpaceBackend(client_factory=lambda: client, poll_interval=0)


def test_status_is_ready_without_network():
    assert make(FakeClient()).status()["state"] == "ready"


def test_run_returns_image_seed_and_rewritten(tmp_path):
    client = FakeClient(FakeJob((output_png(tmp_path), 42, "rich prompt")))
    events = []
    result = make(client).run(req(enhance=True, transparent=True), lambda *e: events.append(e))
    args, api_name = client.calls[0]
    assert api_name == SPACE_API_NAME
    assert args[1] == f"p {config.RGBA_PHRASE}"
    assert result.seed == 42
    assert result.image.size == (40, 20)
    assert result.image.mode == "RGBA"
    assert result.rewritten_prompt == "rich prompt"
    assert events == [("remote-queue", 2, None)]


def test_run_encodes_input_images(tmp_path):
    client = FakeClient(FakeJob(({"path": output_png(tmp_path)}, 1, "")))
    result = make(client).run(req(images=[Image.new("RGB", (8, 8))]), lambda *e: None)
    payload = client.calls[0][0][0]
    assert len(payload) == 1 and "image" in payload[0]
    assert result.rewritten_prompt is None


def test_quota_error_is_friendly():
    client = FakeClient(submit_error=Exception("You have exceeded your GPU quota (60s left)"))
    with pytest.raises(BackendError, match="quota"):
        make(client).run(req(), lambda *e: None)


def test_friendly_error_generic():
    assert friendly_space_error(Exception("boom")) == "HF Space error: boom"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_space_backend.py -q`
Expected: `ModuleNotFoundError: No module named 'app.backends.space'`.

- [ ] **Step 4: Implement `app/backends/space.py`**

```python
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


def encode_images(paths: list[str]):
    if not paths:
        return None
    from gradio_client import handle_file

    return [{"image": handle_file(p), "caption": None} for p in paths]


def build_space_args(req: JobRequest, prompt: str, images_payload) -> list:
    width, height = req.size or _UNUSED_SIZE
    return [
        images_payload,                              # input_images
        prompt,                                      # prompt
        req.enhance,                                 # enable_extend
        req.size is not None,                        # custom_size
        "",                                          # log_dir_input
        req.seed if req.seed is not None else 0,     # seed
        req.seed is None,                            # randomize_seed
        height,                                      # height
        width,                                       # width
        req.negative_prompt,                         # negative_prompt_input
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

    return Client(config.SPACE_ID, hf_token=os.environ.get("HF_TOKEN") or None)


class SpaceBackend:
    name = "space"

    def __init__(self, client_factory=None, poll_interval: float = 1.0):
        self._client_factory = client_factory or _default_client
        self._client = None
        self._poll_interval = poll_interval

    def status(self) -> dict:
        return {"state": "ready", "detail": f"Remote: huggingface.co/spaces/{config.SPACE_ID}"}

    def run(self, req: JobRequest, on_progress: ProgressFn) -> JobResult:
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
                while not job.done():
                    rank = getattr(job.status(), "rank", None)
                    on_progress("remote-queue", rank + 1 if isinstance(rank, int) else None, None)
                    time.sleep(self._poll_interval)
                image_out, seed, rewritten = job.result()
                with Image.open(_result_path(image_out)) as im:
                    im.load()
                    image = im.copy()
            except Exception as e:  # noqa: BLE001 - every remote failure becomes a user message
                raise BackendError(friendly_space_error(e)) from e
        return JobResult(image=image, seed=int(seed), rewritten_prompt=rewritten or None)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```powershell
git add app/backends/space.py tests/test_space_backend.py
git commit -m "feat: HF Space backend via gradio_client"
```

---

### Task 8: Frontend

**Files:**
- Replace: `static/index.html`
- Create: `static/style.css`, `static/app.js`, `run.ps1`

**Interfaces:**
- Consumes:
  - `GET /api/options`, `GET /api/health`, `GET /api/history`
  - `POST /api/jobs` (multipart: `backend, prompt, negative_prompt, size, steps, seed, transparent, enhance, images[]`)
  - `GET /api/jobs/{id}`, `DELETE /api/history/{id}`, `/outputs/<file>`
- Produces: the UI.

- [ ] **Step 1: Write `static/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen-Image Studio</title>
  <link rel="stylesheet" href="style.css">
</head>
<body data-mode="generate" data-backend="local">
  <header class="topbar">
    <h1>Qwen-Image <span>2.1</span></h1>
    <nav class="tabs">
      <button class="tab active" data-tab="generate">Generate</button>
      <button class="tab" data-tab="edit">Edit</button>
      <button class="tab" data-tab="gallery">Gallery</button>
    </nav>
    <div class="backend-switch" role="radiogroup" aria-label="Backend">
      <label><input type="radio" name="backend" value="local" checked> Local GPU <span class="dot" id="dot-local"></span></label>
      <label><input type="radio" name="backend" value="space"> HF Space <span class="dot" id="dot-space"></span></label>
    </div>
  </header>

  <div id="banner" class="banner hidden"></div>

  <main>
    <section id="panel-create" class="create">
      <form id="job-form" class="controls" autocomplete="off">
        <div class="edit-only">
          <div id="dropzone" class="dropzone">
            Drop up to 10 images here or <button type="button" id="pick" class="link">browse</button>
            <input type="file" id="file-input" accept="image/*" multiple hidden>
          </div>
          <ul id="thumbs" class="thumbs"></ul>
        </div>

        <label class="field"><span id="prompt-label">Prompt</span>
          <textarea id="prompt" rows="5" placeholder="A cozy cabin in a snowy forest at dusk, warm light in the windows"></textarea>
        </label>

        <div class="row wrap">
          <label class="toggle"><input type="checkbox" id="enhance"> Enhance prompt</label>
          <label class="toggle"><input type="checkbox" id="transparent"> Transparent background</label>
        </div>

        <details><summary>Negative prompt</summary>
          <textarea id="negative" rows="2" placeholder="blurry, low quality"></textarea>
        </details>

        <div class="field"><span>Size</span>
          <div class="seg" id="tier">
            <button type="button" data-tier="auto">Auto</button>
            <button type="button" data-tier="2K">2K</button>
            <button type="button" data-tier="1K">1K</button>
            <button type="button" data-tier="custom">Custom</button>
          </div>
          <div class="seg" id="ratios"></div>
          <div id="custom-size" class="row hidden">
            <input type="number" id="cw" min="256" max="2688" step="16" value="2048"> ×
            <input type="number" id="ch" min="256" max="2688" step="16" value="2048">
          </div>
        </div>

        <div class="row">
          <label class="field local-only"><span>Steps</span>
            <input type="number" id="steps" min="1" max="100" value="40">
          </label>
          <label class="field grow"><span>Seed</span>
            <span class="row"><input type="number" id="seed" min="0" placeholder="random">
            <button type="button" id="dice" title="Random seed">🎲</button></span>
          </label>
        </div>

        <button type="submit" id="submit" class="primary">Generate</button>
        <p class="hint">Ctrl+Enter to submit</p>
      </form>

      <div class="result">
        <div id="progress" class="progress hidden">
          <div class="bar"><div id="bar-fill"></div></div>
          <span id="progress-text"></span>
        </div>
        <div id="error" class="error hidden"></div>
        <div id="warning" class="warning hidden"></div>
        <div id="result-empty" class="empty">Your image will appear here.</div>
        <figure id="result-figure" class="hidden">
          <div id="result-inputs" class="thumbs small"></div>
          <div id="result-frame" class="frame"><img id="result-img" alt="Result"></div>
          <figcaption>
            <div id="rewritten" class="rewritten hidden">
              <strong>Enhanced prompt</strong>
              <p id="rewritten-text"></p>
              <button type="button" id="use-rewritten">Use as my prompt</button>
            </div>
            <div class="actions">
              <a id="download" class="btn" download>Download</a>
              <button type="button" id="reuse">Reuse settings</button>
              <button type="button" id="edit-this">Edit this image</button>
              <button type="button" id="copy-seed">Copy seed</button>
            </div>
          </figcaption>
        </figure>
      </div>
    </section>

    <section id="panel-gallery" class="gallery hidden">
      <div class="filters">
        <div class="seg" id="filter-mode">
          <button data-v="all" class="active">All</button>
          <button data-v="generate">Generated</button>
          <button data-v="edit">Edited</button>
        </div>
        <div class="seg" id="filter-backend">
          <button data-v="all" class="active">All</button>
          <button data-v="local">Local</button>
          <button data-v="space">Space</button>
        </div>
      </div>
      <div id="grid" class="grid"></div>
      <p id="grid-empty" class="empty hidden">Nothing here yet.</p>
    </section>
  </main>

  <dialog id="detail">
    <div class="detail-body">
      <div id="detail-frame" class="frame"><img id="detail-img" alt=""></div>
      <div class="detail-meta">
        <div id="detail-inputs" class="thumbs small"></div>
        <dl id="detail-dl"></dl>
        <div class="actions">
          <button type="button" id="d-reuse">Reuse</button>
          <button type="button" id="d-edit">Edit this</button>
          <a id="d-download" class="btn" download>Download</a>
          <button type="button" id="d-delete" class="danger">Delete</button>
          <button type="button" id="d-close">Close</button>
        </div>
      </div>
    </div>
  </dialog>

  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write `static/style.css`**

```css
:root {
  --bg: #0f1115; --panel: #171a21; --panel-2: #1f232c; --border: #2a2f3a;
  --text: #e6e8ee; --muted: #8b93a7; --accent: #7c5cff; --accent-2: #9d85ff;
  --ok: #3ecf8e; --warn: #f5a524; --err: #ff5d5d; --radius: 10px;
  color-scheme: dark;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
button, input, textarea { font: inherit; color: inherit; }
button, .btn { background: var(--panel-2); border: 1px solid var(--border); border-radius: 8px;
  padding: 6px 12px; cursor: pointer; text-decoration: none; display: inline-block; }
button:hover, .btn:hover { border-color: var(--accent); }
button:disabled { opacity: .5; cursor: not-allowed; }
button.primary { background: var(--accent); border-color: var(--accent); font-weight: 600; padding: 10px; width: 100%; }
button.primary:hover:not(:disabled) { background: var(--accent-2); }
button.danger { color: var(--err); }
button.link { background: none; border: none; color: var(--accent-2); padding: 0; text-decoration: underline; }
input[type=number], textarea { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 8px; width: 100%; }
textarea { resize: vertical; }
input:focus, textarea:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
.hidden { display: none !important; }

.topbar { display: flex; align-items: center; gap: 24px; padding: 12px 20px;
  border-bottom: 1px solid var(--border); background: var(--panel); position: sticky; top: 0; z-index: 5; }
.topbar h1 { font-size: 18px; margin: 0; }
.topbar h1 span { color: var(--accent-2); }
.tabs { display: flex; gap: 4px; }
.tab { background: none; border-color: transparent; }
.tab.active { background: var(--panel-2); border-color: var(--border); }
.backend-switch { margin-left: auto; display: flex; gap: 16px; }
.backend-switch label { display: flex; align-items: center; gap: 6px; cursor: pointer; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }
.dot.ready { background: var(--ok); } .dot.loading { background: var(--warn); } .dot.error { background: var(--err); }

.banner { padding: 8px 20px; background: #2a2110; color: var(--warn); border-bottom: 1px solid var(--border); }
.banner.error { background: #2a1414; color: var(--err); }

main { padding: 20px; }
.create { display: grid; grid-template-columns: 380px 1fr; gap: 20px; align-items: start; }
@media (max-width: 900px) { .create { grid-template-columns: 1fr; } .topbar { flex-wrap: wrap; } }
.controls { background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 16px; display: flex; flex-direction: column; gap: 14px; }
.field { display: flex; flex-direction: column; gap: 6px; }
.field > span { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
.row { display: flex; gap: 8px; align-items: center; }
.row.wrap { flex-wrap: wrap; gap: 16px; }
.grow { flex: 1; }
.toggle { display: flex; gap: 6px; align-items: center; cursor: pointer; }
details summary { cursor: pointer; color: var(--muted); margin-bottom: 6px; }
.hint { margin: -6px 0 0; color: var(--muted); font-size: 12px; text-align: center; }
.seg { display: flex; flex-wrap: wrap; gap: 4px; }
.seg button { padding: 4px 10px; }
.seg button.active { background: var(--accent); border-color: var(--accent); }
#ratios:empty { display: none; }

body[data-mode="generate"] .edit-only { display: none; }
body[data-backend="space"] .local-only { display: none; }

.dropzone { border: 2px dashed var(--border); border-radius: var(--radius); padding: 18px; text-align: center; color: var(--muted); }
.dropzone.over { border-color: var(--accent); color: var(--text); }
.thumbs { list-style: none; margin: 8px 0 0; padding: 0; display: flex; flex-wrap: wrap; gap: 8px; }
.thumbs li { position: relative; width: 72px; }
.thumbs img { width: 72px; height: 72px; object-fit: cover; border-radius: 6px; border: 1px solid var(--border); display: block; }
.thumbs li .tools { display: flex; justify-content: space-between; margin-top: 2px; }
.thumbs li .tools button { padding: 0 6px; font-size: 12px; }
.thumbs.small img { width: 48px; height: 48px; }
.thumbs .num { position: absolute; top: 2px; left: 4px; font-size: 11px; background: #000a; padding: 0 4px; border-radius: 4px; }

.result { min-height: 60vh; display: flex; flex-direction: column; gap: 12px; }
.empty { color: var(--muted); border: 1px dashed var(--border); border-radius: var(--radius); padding: 60px; text-align: center; }
.progress { display: flex; align-items: center; gap: 12px; }
.bar { flex: 1; height: 6px; background: var(--panel-2); border-radius: 3px; overflow: hidden; }
#bar-fill { height: 100%; width: 0; background: var(--accent); transition: width .3s; }
.bar.indeterminate #bar-fill { width: 30%; animation: slide 1.2s infinite ease-in-out; }
@keyframes slide { from { transform: translateX(-100%); } to { transform: translateX(350%); } }
.error { color: var(--err); background: #2a1414; padding: 10px; border-radius: 8px; }
.warning { color: var(--warn); background: #2a2110; padding: 10px; border-radius: 8px; }
figure { margin: 0; display: flex; flex-direction: column; gap: 12px; }
.frame { border-radius: var(--radius); overflow: hidden; border: 1px solid var(--border); background: var(--panel); display: flex; justify-content: center; }
.frame img { max-width: 100%; max-height: 75vh; display: block; }
.frame.checker { background-color: #fff;
  background-image: linear-gradient(45deg, #ccc 25%, transparent 25%), linear-gradient(-45deg, #ccc 25%, transparent 25%),
    linear-gradient(45deg, transparent 75%, #ccc 75%), linear-gradient(-45deg, transparent 75%, #ccc 75%);
  background-size: 20px 20px; background-position: 0 0, 0 10px, 10px -10px, -10px 0; }
.rewritten { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 10px; }
.rewritten p { margin: 6px 0; white-space: pre-wrap; }
.actions { display: flex; flex-wrap: wrap; gap: 8px; }

.filters { display: flex; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }
.card { cursor: pointer; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--panel); }
.card img { width: 100%; aspect-ratio: 1; object-fit: cover; display: block; }
.card p { margin: 0; padding: 6px 8px; font-size: 12px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

dialog { background: var(--panel); color: var(--text); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px; max-width: 1100px; width: 95vw; }
dialog::backdrop { background: #000b; }
.detail-body { display: grid; grid-template-columns: 1fr 320px; gap: 16px; }
@media (max-width: 900px) { .detail-body { grid-template-columns: 1fr; } }
.detail-meta dl { display: grid; grid-template-columns: auto 1fr; gap: 4px 12px; margin: 12px 0; }
.detail-meta dt { color: var(--muted); }
.detail-meta dd { margin: 0; word-break: break-word; white-space: pre-wrap; }
```

- [ ] **Step 3: Write `static/app.js`**

```js
"use strict";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  mode: "generate",
  options: null,
  health: {},
  sizeByMode: { generate: { tier: "2K", ratio: "1:1" }, edit: { tier: "auto", ratio: "1:1" } },
  inputs: [],            // [{ blob, url }]
  lastResult: null,
  pollTimer: null,
  busy: false,
  history: [],
  filters: { mode: "all", backend: "all" },
  detail: null,
};

// ---------- helpers ----------
async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch { /* keep status text */ }
    throw new Error(msg);
  }
  return res.status === 204 ? null : res.json();
}

const backend = () => $('input[name="backend"]:checked').value;
const outputUrl = (file) => `/outputs/${encodeURIComponent(file)}`;
const round16 = (v) => Math.round(v / 16) * 16;
const show = (el, on = true) => el.classList.toggle("hidden", !on);

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "onclick") node.addEventListener("click", v);
    else if (k === "class") node.className = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c);
  return node;
}

// ---------- tabs & mode ----------
function switchTab(tab) {
  $$(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  const gallery = tab === "gallery";
  show($("#panel-create"), !gallery);
  show($("#panel-gallery"), gallery);
  if (gallery) loadHistory();
  else setMode(tab);
}

function setMode(mode) {
  state.mode = mode;
  document.body.dataset.mode = mode;
  $("#prompt-label").textContent = mode === "edit" ? "Instruction" : "Prompt";
  $("#prompt").placeholder = mode === "edit"
    ? "Change the background to a sunset beach  ·  Extract the person  ·  Put the cat from image 1 on the sofa in image 2"
    : "A cozy cabin in a snowy forest at dusk, warm light in the windows";
  $("#submit").textContent = mode === "edit" ? "Edit" : "Generate";
  renderSize();
  updateSubmitEnabled();
}

// ---------- size picker ----------
function renderSize() {
  const sel = state.sizeByMode[state.mode];
  $$("#tier button").forEach((b) => b.classList.toggle("active", b.dataset.tier === sel.tier));
  const ratios = $("#ratios");
  ratios.innerHTML = "";
  if (state.options && (sel.tier === "2K" || sel.tier === "1K")) {
    for (const p of state.options.sizes[sel.tier]) {
      const b = el("button", { type: "button", title: `${p.width}×${p.height}` }, p.label);
      b.classList.toggle("active", p.label === sel.ratio);
      b.addEventListener("click", () => { sel.ratio = p.label; renderSize(); });
      ratios.append(b);
    }
  }
  show($("#custom-size"), sel.tier === "custom");
}

function currentSize() {
  const sel = state.sizeByMode[state.mode];
  if (sel.tier === "auto") return "auto";
  if (sel.tier === "custom") return `${round16(+$("#cw").value)}x${round16(+$("#ch").value)}`;
  const p = state.options.sizes[sel.tier].find((x) => x.label === sel.ratio) || state.options.sizes[sel.tier][0];
  return `${p.width}x${p.height}`;
}

function applySize(mode, width, height) {
  const sel = state.sizeByMode[mode];
  for (const [tier, presets] of Object.entries(state.options.sizes)) {
    const p = presets.find((x) => x.width === width && x.height === height);
    if (p) { sel.tier = tier; sel.ratio = p.label; return; }
  }
  sel.tier = "custom";
  $("#cw").value = width;
  $("#ch").value = height;
}

// ---------- edit inputs ----------
function addBlobs(blobs) {
  const max = state.options ? state.options.limits.max_images : 10;
  for (const blob of blobs) {
    if (!blob.type.startsWith("image/")) continue;
    if (state.inputs.length >= max) { showError(`At most ${max} input images.`); break; }
    state.inputs.push({ blob, url: URL.createObjectURL(blob) });
  }
  renderThumbs();
}

async function addFromUrl(url) {
  const blob = await (await fetch(url)).blob();
  addBlobs([blob]);
}

function moveInput(i, delta) {
  const j = i + delta;
  if (j < 0 || j >= state.inputs.length) return;
  [state.inputs[i], state.inputs[j]] = [state.inputs[j], state.inputs[i]];
  renderThumbs();
}

function removeInput(i) {
  URL.revokeObjectURL(state.inputs[i].url);
  state.inputs.splice(i, 1);
  renderThumbs();
}

function renderThumbs() {
  const ul = $("#thumbs");
  ul.innerHTML = "";
  state.inputs.forEach((inp, i) => {
    ul.append(el("li", {},
      el("span", { class: "num" }, String(i + 1)),
      el("img", { src: inp.url, alt: `Input ${i + 1}` }),
      el("div", { class: "tools" },
        el("button", { type: "button", title: "Move left", onclick: () => moveInput(i, -1) }, "←"),
        el("button", { type: "button", title: "Remove", onclick: () => removeInput(i) }, "×"),
        el("button", { type: "button", title: "Move right", onclick: () => moveInput(i, 1) }, "→"))));
  });
  updateSubmitEnabled();
}

// ---------- health ----------
async function refreshHealth() {
  try {
    state.health = (await api("/api/health")).backends;
  } catch {
    state.health = {};
  }
  for (const name of ["local", "space"]) {
    const s = state.health[name];
    $(`#dot-${name}`).className = `dot ${s ? s.state : ""}`;
    $(`#dot-${name}`).title = s ? `${s.state}: ${s.detail}` : "unreachable";
  }
  const s = state.health[backend()];
  const banner = $("#banner");
  if (s && s.state !== "ready") {
    banner.textContent = s.state === "loading"
      ? "Loading Qwen-Image-2.1 on your GPU… (the first run downloads the weights). You can use HF Space meanwhile."
      : `${backend() === "local" ? "Local GPU" : "HF Space"} unavailable — ${s.detail}`;
    banner.classList.toggle("error", s.state === "error");
    show(banner, true);
  } else {
    show(banner, false);
  }
  updateSubmitEnabled();
}

function updateSubmitEnabled() {
  const s = state.health[backend()];
  const ready = s && s.state === "ready";
  const needsInputs = state.mode === "edit" && state.inputs.length === 0;
  $("#submit").disabled = state.busy || !ready || needsInputs;
}

// ---------- messages ----------
function showError(msg) { $("#error").textContent = msg; show($("#error"), true); }
function showWarning(msg) { $("#warning").textContent = msg; show($("#warning"), !!msg); }
function clearMessages() { show($("#error"), false); show($("#warning"), false); }

// ---------- jobs ----------
async function submitJob(ev) {
  ev.preventDefault();
  if ($("#submit").disabled) return;
  const prompt = $("#prompt").value.trim();
  if (!prompt) return showError("Please enter a prompt.");

  const fd = new FormData();
  fd.append("backend", backend());
  fd.append("prompt", prompt);
  const neg = $("#negative").value;
  if (neg.trim()) fd.append("negative_prompt", neg);
  fd.append("size", currentSize());
  if (backend() === "local") fd.append("steps", $("#steps").value || "40");
  const seed = $("#seed").value.trim();
  if (seed) fd.append("seed", seed);
  fd.append("transparent", $("#transparent").checked ? "true" : "false");
  fd.append("enhance", $("#enhance").checked ? "true" : "false");
  if (state.mode === "edit") {
    state.inputs.forEach((inp, i) => fd.append("images", inp.blob, `input-${i}.png`));
  }

  clearMessages();
  setBusy(true);
  setProgress({ status: "queued", queue_position: null });
  try {
    const { job_id } = await api("/api/jobs", { method: "POST", body: fd });
    pollJob(job_id);
  } catch (err) {
    setBusy(false);
    show($("#progress"), false);
    showError(err.message);
  }
}

function setBusy(busy) {
  state.busy = busy;
  updateSubmitEnabled();
}

function setProgress(job) {
  show($("#progress"), true);
  const bar = $(".bar");
  const fill = $("#bar-fill");
  let text = "Working…";
  let pct = null;
  if (job.status === "queued") {
    text = job.queue_position ? `Queued (#${job.queue_position})` : "Queued…";
  } else if (job.phase === "enhancing") {
    text = "Enhancing prompt…";
  } else if (job.phase === "generating" && job.total_steps) {
    text = `Step ${job.step}/${job.total_steps}`;
    pct = (100 * job.step) / job.total_steps;
  } else if (job.phase === "remote-queue") {
    text = job.step ? `Waiting on HF Space (#${job.step})` : "Running on HF Space…";
  }
  $("#progress-text").textContent = text;
  bar.classList.toggle("indeterminate", pct === null);
  fill.style.width = pct === null ? "" : `${pct}%`;
}

function pollJob(jobId) {
  clearInterval(state.pollTimer);
  state.pollTimer = setInterval(async () => {
    let job;
    try {
      job = await api(`/api/jobs/${jobId}`);
    } catch (err) {
      clearInterval(state.pollTimer);
      setBusy(false);
      show($("#progress"), false);
      return showError(err.message);
    }
    if (job.status === "done") {
      clearInterval(state.pollTimer);
      setBusy(false);
      show($("#progress"), false);
      showResult(job.result);
      showWarning(job.warning);
    } else if (job.status === "error") {
      clearInterval(state.pollTimer);
      setBusy(false);
      show($("#progress"), false);
      showError(job.error);
    } else {
      setProgress(job);
    }
  }, 1000);
}

function showResult(meta) {
  state.lastResult = meta;
  show($("#result-empty"), false);
  show($("#result-figure"), true);
  $("#result-img").src = `${outputUrl(meta.file)}?t=${Date.now()}`;
  $("#result-frame").classList.toggle("checker", meta.transparent);
  $("#download").href = outputUrl(meta.file);
  $("#download").download = meta.file;
  const inputs = $("#result-inputs");
  inputs.innerHTML = "";
  meta.input_files.forEach((f) => inputs.append(el("img", { src: outputUrl(f), alt: "Input" })));
  show($("#rewritten"), !!meta.rewritten_prompt);
  $("#rewritten-text").textContent = meta.rewritten_prompt || "";
}

// ---------- reuse / edit this ----------
async function reuseSettings(meta) {
  switchTab(meta.mode);
  $(`input[name="backend"][value="${meta.backend}"]`).checked = true;
  onBackendChange();
  $("#prompt").value = meta.prompt;
  $("#negative").value = meta.negative_prompt.trim();
  $("#seed").value = meta.seed;
  if (meta.steps) $("#steps").value = meta.steps;
  $("#transparent").checked = meta.transparent;
  $("#enhance").checked = meta.enhance;
  applySize(meta.mode, meta.width, meta.height);
  if (meta.mode === "edit") {
    state.inputs.forEach((inp) => URL.revokeObjectURL(inp.url));
    state.inputs = [];
    for (const f of meta.input_files) await addFromUrl(outputUrl(f));
  }
  renderSize();
  renderThumbs();
}

async function editThis(meta) {
  switchTab("edit");
  await addFromUrl(outputUrl(meta.file));
}

// ---------- gallery ----------
async function loadHistory() {
  try {
    state.history = await api("/api/history");
  } catch (err) {
    state.history = [];
  }
  renderGrid();
}

function renderGrid() {
  const grid = $("#grid");
  grid.innerHTML = "";
  const items = state.history.filter((m) =>
    (state.filters.mode === "all" || m.mode === state.filters.mode) &&
    (state.filters.backend === "all" || m.backend === state.filters.backend));
  for (const m of items) {
    grid.append(el("div", { class: "card", onclick: () => openDetail(m) },
      el("img", { src: outputUrl(m.file), loading: "lazy", alt: m.prompt }),
      el("p", { title: m.prompt }, `${m.mode === "edit" ? "✎ " : ""}${m.prompt}`)));
  }
  show($("#grid-empty"), items.length === 0);
}

function openDetail(meta) {
  state.detail = meta;
  $("#detail-img").src = outputUrl(meta.file);
  $("#detail-frame").classList.toggle("checker", meta.transparent);
  $("#d-download").href = outputUrl(meta.file);
  $("#d-download").download = meta.file;
  const inputs = $("#detail-inputs");
  inputs.innerHTML = "";
  meta.input_files.forEach((f) => inputs.append(el("img", { src: outputUrl(f), alt: "Input" })));
  const rows = [
    ["Prompt", meta.prompt],
    ["Enhanced", meta.rewritten_prompt],
    ["Negative", meta.negative_prompt.trim()],
    ["Mode", meta.mode],
    ["Backend", meta.backend],
    ["Size", `${meta.width}×${meta.height} (${meta.size_label})`],
    ["Steps", meta.steps],
    ["Seed", meta.seed],
    ["Transparent", meta.transparent ? "yes" : "no"],
    ["Time", `${meta.duration_s}s`],
    ["Created", new Date(meta.created_at).toLocaleString()],
  ];
  const dl = $("#detail-dl");
  dl.innerHTML = "";
  for (const [k, v] of rows) {
    if (v === null || v === undefined || v === "") continue;
    dl.append(el("dt", {}, k), el("dd", {}, String(v)));
  }
  $("#detail").showModal();
}

async function deleteDetail() {
  const meta = state.detail;
  if (!meta || !confirm("Delete this image?")) return;
  await api(`/api/history/${meta.id}`, { method: "DELETE" });
  $("#detail").close();
  loadHistory();
}

// ---------- backend switch ----------
function onBackendChange() {
  document.body.dataset.backend = backend();
  refreshHealth();
}

// ---------- wiring ----------
function wire() {
  $$(".tab").forEach((b) => b.addEventListener("click", () => switchTab(b.dataset.tab)));
  $$('input[name="backend"]').forEach((r) => r.addEventListener("change", onBackendChange));
  $$("#tier button").forEach((b) => b.addEventListener("click", () => {
    state.sizeByMode[state.mode].tier = b.dataset.tier;
    renderSize();
  }));
  $("#job-form").addEventListener("submit", submitJob);
  $("#prompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) submitJob(e);
  });
  $("#dice").addEventListener("click", () => { $("#seed").value = Math.floor(Math.random() * 2 ** 32); });

  const dz = $("#dropzone");
  $("#pick").addEventListener("click", () => $("#file-input").click());
  $("#file-input").addEventListener("change", (e) => { addBlobs(e.target.files); e.target.value = ""; });
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("over"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("over"));
  dz.addEventListener("drop", (e) => { e.preventDefault(); dz.classList.remove("over"); addBlobs(e.dataTransfer.files); });
  document.addEventListener("paste", (e) => {
    if (state.mode !== "edit") return;
    const files = Array.from(e.clipboardData.files || []);
    if (files.length) addBlobs(files);
  });

  $("#use-rewritten").addEventListener("click", () => {
    $("#prompt").value = state.lastResult.rewritten_prompt;
    $("#enhance").checked = false;
  });
  $("#reuse").addEventListener("click", () => reuseSettings(state.lastResult));
  $("#edit-this").addEventListener("click", () => editThis(state.lastResult));
  $("#copy-seed").addEventListener("click", () => navigator.clipboard.writeText(String(state.lastResult.seed)));

  $$("#filter-mode button, #filter-backend button").forEach((b) => b.addEventListener("click", () => {
    const key = b.parentElement.id === "filter-mode" ? "mode" : "backend";
    state.filters[key] = b.dataset.v;
    $$(`#${b.parentElement.id} button`).forEach((x) => x.classList.toggle("active", x === b));
    renderGrid();
  }));
  $("#d-close").addEventListener("click", () => $("#detail").close());
  $("#d-reuse").addEventListener("click", () => { $("#detail").close(); reuseSettings(state.detail); });
  $("#d-edit").addEventListener("click", () => { $("#detail").close(); editThis(state.detail); });
  $("#d-delete").addEventListener("click", deleteDetail);
}

async function init() {
  wire();
  state.options = await api("/api/options");
  setMode("generate");
  await refreshHealth();
  setInterval(refreshHealth, 3000);
}

init().catch((err) => showError(`Failed to start: ${err.message}`));
```

- [ ] **Step 4: Write `run.ps1`**

```powershell
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
& .\.venv\Scripts\python.exe -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
```

- [ ] **Step 5: Smoke-test the UI with the local backend disabled** (no model download)

```powershell
$env:QWEN_DISABLE_LOCAL = "1"; .\run.ps1
```

Open `http://127.0.0.1:8000` and check each of these:
- The header shows Local with a grey dot, and a banner says "Local GPU unavailable — Disabled…".
- After switching to HF Space, the dot turns green and the Steps field is hidden.
- The Generate tab accepts the prompt "a red fox in the snow" with size 1K 1:1. After Generate, the progress shows "Waiting on HF Space", then the image appears.
- In the Gallery, the image is listed, the detail view opens, and Reuse fills in the form.
- Edit this → Edit tab shows the image as input 1. The instruction "make it night time" produces an edited image, with input thumbnails shown above the result.
- Delete in the detail view removes the image.

If the Space call fails with an argument or format error, re-check Task 7 Step 1 (`view_api`) and fix `build_space_args` or `encode_images`.

- [ ] **Step 6: Run the full test suite, then commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
git add static run.ps1
git commit -m "feat: single-page UI for generate, edit and gallery"
```

---

### Task 9: Real GPU verification and README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Start with the local backend enabled**

```powershell
Remove-Item Env:QWEN_DISABLE_LOCAL -ErrorAction SilentlyContinue; .\run.ps1
```

Expected:
- The first run downloads `Qwen/Qwen-Image-2.1`.
- The Local dot is amber while loading and green when ready. Its tooltip says "Loaded on GPU" or "Loaded with CPU offload".
- If loading fails with a gated or 401 error, run `.\.venv\Scripts\huggingface-cli.exe login` and restart.

- [ ] **Step 2: Manual checks on the Local backend**

1. Generate "a lighthouse on a cliff at sunset, oil painting" at 2K 16:9 with 40 steps. Expected: the step counter goes up to 40/40, and the result is 2688×1536.
2. Turn on Transparent and generate "a cute cartoon robot mascot" at 1K 1:1. Expected: a checkerboard background shows through. Confirm the saved file has alpha:
   `.\.venv\Scripts\python.exe -c "from PIL import Image; import glob; f=sorted(glob.glob('outputs/*.png'))[-1]; im=Image.open(f); print(f, im.mode, im.getextrema()[-1] if im.mode=='RGBA' else '')"`
   Expected: `RGBA`, with an alpha range that includes 0. If the mode is `RGB`, check the pipeline for an RGBA output option (`inspect.signature(QwenImage21Pipeline.__call__)`, e.g. `output_type`), apply it in `LocalBackend.run` when `req.transparent`, and add a test.
3. Edit with two input images and the instruction "put the robot from image 1 next to the lighthouse in image 2". Expected: the result combines both images.
4. Generate with Enhance prompt on, using "cat astronaut". Expected:
   - "Enhancing prompt…" appears, then the first run downloads PE-T2I.
   - The enhanced prompt is shown and the image matches it.
   - The Local tooltip then says "CPU offload (prompt enhancer in use)".
5. In the Gallery, all results appear, and the Local/Space filters work.

- [ ] **Step 3: Write `README.md`**

````markdown
# Qwen-Image Studio

A local web UI for [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1). It supports:
- text-to-image
- editing with up to 10 reference images
- transparent (RGBA) output
- prompt enhancement
- a history gallery

Each job runs on your own GPU or on the official Hugging Face Space.

## Setup

```powershell
python -m pip install --user uv
python -m uv venv --python 3.12 .venv
python -m uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu128
python -m uv pip install --python .venv -r requirements.txt
```

## Run

```powershell
.\run.ps1          # http://127.0.0.1:8000
```

- The first local run downloads about 14 GB of weights. The prompt enhancers are about 18 GB each and download on first use.
- `$env:QWEN_DISABLE_LOCAL = "1"` runs with the HF Space backend only.
- `$env:HF_TOKEN = "hf_..."` gives a larger Space quota and access to gated downloads.

Results are saved in `outputs/`, one PNG and one JSON file per image.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
````

- [ ] **Step 4: Commit**

```powershell
git add README.md app static tests
git commit -m "docs: README; verified on local GPU"
```
