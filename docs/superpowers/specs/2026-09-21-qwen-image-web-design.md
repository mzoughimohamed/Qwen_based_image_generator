# Qwen-Image Web — Design

Date: 2026-09-21
Status: Approved in brainstorming (revised after reviewing the official HF Space), pending spec review

## Goal

A small local website for a single user at `http://127.0.0.1` that serves **Qwen-Image-2.1**.
The model runs either on the user's own GPU (RTX 5000 Ada, 32 GB) using the open weights
(`Qwen/Qwen-Image-2.1`, Diffusers `QwenImage21Pipeline`), or remotely through the official
Hugging Face Space (`Qwen/Qwen-Image-2.1`). It supports:

1. Text-to-image generation
2. Image editing with 1–10 input images (single-image edits and multi-reference composition)
3. Transparent (RGBA) output, including subject extraction
4. Optional prompt enhancement, showing the rewritten prompt
5. A history gallery of all results
6. A per-job choice of backend: **Local GPU** (default) or **HF Space**

Out of scope: multi-user access, authentication, DashScope API keys, LoRAs, ComfyUI.

## Architecture

FastAPI backend with a static HTML/CSS/JS frontend served by the same process.
No Node, no build step, no database.

```
image_generator/
  .venv/                 # Python 3.12 via uv (PyTorch wheels may not exist for 3.14)
  app/
    main.py              # FastAPI app: API routes, serves static/ and outputs/
    config.py            # size presets, limits, paths, RGBA phrase
    schemas.py           # JobRequest / JobResult dataclasses + validation
    backends/
      base.py            # Backend protocol
      local.py           # LocalBackend: QwenImage21Pipeline + prompt enhancer
      enhancer.py        # PromptEnhancer: PE-T2I / PE-I2I, load-on-demand
      space.py           # SpaceBackend: gradio_client -> HF Space
    jobs.py              # single-worker FIFO job queue
    storage.py           # PNG + JSON metadata persistence, history listing
  static/
    index.html, app.js, style.css
  outputs/               # <id>.png, <id>.json, <id>.input-<n>.png for edits
  tests/
  requirements.txt
  run.ps1
```

### Request / result model (`app/schemas.py`)

```
JobRequest:
  backend: "local" | "space"
  prompt: str                   # non-empty
  negative_prompt: str = " "
  images: list[PIL.Image]       # 0 = text-to-image, 1..10 = edit
  size: (w, h) | None           # None = auto (edits: fit to the first image's aspect ratio; T2I: 2048x2048)
  steps: int = 40               # 1..100, local only (the Space controls its own steps)
  seed: int | None              # None = random; the resolved seed is always recorded
  transparent: bool = False
  enhance: bool = False

JobResult:
  image: PIL.Image
  seed: int
  rewritten_prompt: str | None
```

`mode` is derived: `"edit"` if `images` is non-empty, else `"generate"`.

### Backend protocol (`app/backends/base.py`)

```
class Backend(Protocol):
    name: str
    def status(self) -> dict            # {"state": "loading"|"ready"|"error"|"disabled", "detail": str}
    def run(self, req: JobRequest, on_progress: Callable[[str, int | None, int | None], None]) -> JobResult
```

`on_progress(phase, step, total)` reports `phase` ∈ `"enhancing" | "generating" | "remote-queue"`.
Tests inject a `FakeBackend`.

### Local backend (`app/backends/local.py`)

- Loads `Qwen/Qwen-Image-2.1` in `torch.bfloat16` in a background thread when the server starts.
  It tries `.to("cuda")` and, on OOM, reloads with `enable_model_cpu_offload()`.
- Call: `pipe(prompt=..., image=images or None, width=, height=, num_inference_steps=, generator=, callback_on_step_end=...)`.
  - `image` receives the list of RGB PIL images (up to 10).
  - `negative_prompt` is passed only if `inspect.signature` shows the pipeline accepts it.
- Transparent: appends `"This is an RGBA image with transparency. The image has alpha channel and the background is transparent."`
  to the (possibly enhanced) prompt. The result is saved as an RGBA PNG.
- Enhance: runs `PromptEnhancer` first (phase `"enhancing"`), then generates with the rewritten prompt.
  If the enhancer returns a `wh_ratio` and `size` is auto, the enhancer's ratio chooses the preset.

### Prompt enhancer (`app/backends/enhancer.py`)

- Models: `Qwen/Qwen-Image-2.1-PE-T2I` for text-to-image and `Qwen/Qwen-Image-2.1-PE-I2I` for edits.
  Each is a 9B Qwen3.5-VL model of about 18 GB in bf16.
- **Memory strategy:** the enhancer and the image model do not fit in 32 GB together.
  - Enhancers load lazily on first use.
  - Before an enhancer runs, the image pipeline is switched to model CPU offload (and it stays in offload mode after that).
  - An enhancer is moved to CPU after each use, and only one enhancer stays resident in CPU RAM; switching modes frees the other.
  - Model loading goes through a `load_fn` seam, so this logic can be tested without real weights.
- Uses the model card's system prompt (`system_prompt.txt` from the repo) and chat template with
  `enable_thinking=True`, and the sampling settings `temperature=1.0, top_p=0.95, top_k=20`.
  Output tokens are capped at 4096 (the card suggests 16256; this lower cap is a latency trade-off that can be raised in config).
  - For I2I, the input images are included in the user message.
- It parses the last JSON object `{"rewritten_prompt", "wh_ratio"}` after the thinking block.
  If the output can't be parsed, the job uses the original prompt and records a warning.

### Space backend (`app/backends/space.py`)

- `gradio_client.Client("Qwen/Qwen-Image-2.1", hf_token=os.environ.get("HF_TOKEN"))` is created lazily.
- It calls the Space's `generate_with_enhance` endpoint with these arguments, in this order:
  1. `input_images`
  2. `prompt`
  3. `enable_extend`
  4. `custom_size`
  5. `log_dir_input`
  6. `seed`
  7. `randomize_seed`
  8. `height`
  9. `width`
  10. `negative_prompt_input`

  It receives `(image, seed, rewritten_prompt)`.
  - The exact `api_name` and the Gallery input encoding are confirmed with `client.view_api()` during implementation.
  - `custom_size=False` when the size is auto.
- Transparent adds the same RGBA phrase to the prompt. The returned image is saved as-is (RGBA).
- `steps` does not apply. Progress reports the phase `"remote-queue"` with the queue position when gradio_client provides it.
- Errors (quota, Space sleeping or changed, network) become a readable job error. The local backend is unaffected.
- `status()` returns `"ready"` without contacting the network. The first call surfaces any connection problem.

### Size presets (`app/config.py`, matching the Space)

| Tier | Sizes |
|---|---|
| 2K | 2688×1536 (16:9), 1536×2688 (9:16), 2048×2048 (1:1), 2368×1728 (4:3), 1728×2368 (3:4) |
| 1K | 1344×768 (16:9), 768×1344 (9:16), 1024×1024 (1:1), 1184×864 (4:3), 864×1184 (3:4) |

- There is also a **Custom** option: width and height from 256 to 2688, rounded to a multiple of 16.
- The UI offers **Auto** (the default for edits), a 1K/2K tier switch with aspect buttons, and Custom.
- Auto for a local edit chooses the 2K preset whose aspect ratio is nearest to the first input image's.
- Auto for the Space backend sends no size.

### Jobs (`app/jobs.py`)

- One worker thread consumes a FIFO queue, so only one job runs at a time, including Space jobs.
- Job fields:
  - `id, backend, status`, where `status` moves through `queued → running → done | error`
  - `phase, step, total_steps, queue_position, result_id, rewritten_prompt, warning, error`
- Jobs are kept in memory (lost on restart). Results are saved to disk.

### Storage (`app/storage.py`)

- Result id: `YYYYMMDD-HHMMSS-<4 hex>`.
- Files: `<id>.png`, `<id>.json`, plus `<id>.input-<n>.png` for each edit input.
- Metadata fields:
  - `id, mode, backend`
  - `prompt, rewritten_prompt, negative_prompt`
  - `width, height, size_label, steps, seed`
  - `transparent, enhance`
  - `input_files[], duration_s, created_at`
- `list_history()` returns metadata newest-first. `delete(id)` removes all files for that id.

### API (`app/main.py`)

| Method | Path | Body / Response |
|---|---|---|
| GET | `/api/health` | `{backends: {local: {state, detail, offload}, space: {state, detail}}}` |
| GET | `/api/options` | size presets, limits, backend names |
| POST | `/api/jobs` | multipart form: `backend, prompt, negative_prompt?, size? ("auto"\|"2048x2048"\|…), steps?, seed?, transparent, enhance`, `images[]` 0–10 files, or `from_result` ids to reuse gallery images as inputs → `{job_id}` |
| GET | `/api/jobs/{id}` | job state |
| GET | `/api/history` | list of metadata |
| DELETE | `/api/history/{id}` | 204 |
| GET | `/outputs/...` | static result files |
| GET | `/` | `static/index.html` |

A single `/api/jobs` endpoint serves both modes, as the Space does. The server binds to `127.0.0.1` only.

## Frontend

A single page with a dark theme and three tabs, written in plain HTML/CSS/JS. The **backend switch** (Local GPU / HF Space)
sits in the header and shows each backend's status.

- **Generate tab:**
  - prompt textarea, with Ctrl+Enter to submit
  - "Enhance prompt" toggle
  - collapsible negative prompt
  - size picker: tier + aspect / Custom
  - Transparent toggle
  - steps field (hidden for the Space backend)
  - seed field with a random (🎲) button
  - Generate button
- **Edit tab:**
  - a drop zone that accepts up to 10 images, shown as reorderable thumbnails with remove (×) buttons
  - instruction textarea
  - Enhance, size (defaults to Auto), steps, seed and Transparent controls
  - Edit button
- **Result panel** (shared by Generate and Edit):
  - progress: "Queued (#2)" / "Enhancing prompt…" / "Step 12/40" / "Waiting on HF Space (#5)"
  - the result on a checkerboard background when transparent
  - the rewritten prompt when enhancement was used, with a "Use as my prompt" button
  - input thumbnails next to the result for edits
  - actions: Download, Reuse settings, Edit this image (adds it to the Edit inputs), Copy seed
- **Gallery tab:**
  - thumbnail grid, newest first
  - filters: All / Generated / Edited and Local / Space
  - clicking a thumbnail opens a detail view with the prompts, settings and inputs, plus Reuse, Edit this and Delete
- The Local Generate/Edit buttons are disabled while the local backend is loading. Space jobs work regardless.
- The frontend polls `/api/jobs/{id}` every 1 s.

## Error handling

- **Local backend not ready or in error:** a local job submission returns 503, with the state and detail.
- **CUDA OOM:** the job fails with a readable message and `torch.cuda.empty_cache()` runs.
  - If it happens during enhancement, the enhancer is unloaded before the job fails.
  - The server stays up.
- **Enhancer output can't be parsed:** the job falls back to the original prompt and sets `warning`.
- **Space failures:** the job fails with the gradio_client message, simplified where possible ("Space quota exceeded — set HF_TOKEN or use Local").
- **Invalid input:** the server returns 400 for any of these:
  - an empty prompt
  - an unknown size or backend
  - steps out of range
  - more than 10 images
  - a non-image upload
  - an upload larger than 20 MB
  - an unknown `from_result` id
- **Model downloads:** the first local use downloads the weights, about 14+ GB for the image model and about 18 GB per enhancer.
  If access is gated, the user runs `huggingface-cli login`.

## Testing

- pytest with a `FakeBackend`, so no GPU or network is needed. Tests cover:
  - validation of every `/api/jobs` field, including the 10-image limit and `from_result`
  - the job lifecycle, FIFO ordering, progress phases, and error propagation
  - the storage metadata round-trip, including input files, delete, and RGBA preservation
  - size resolution: auto for an edit picks the nearest aspect ratio, and custom rounds to a multiple of 16
  - enhancer JSON parsing, covering thinking blocks, trailing text and malformed output (pure function)
  - enhancer memory orchestration with fake `load_fn` models: lazy load, one resident, unload on error
  - Space argument mapping (pure function from `JobRequest` to the ordered args), with gradio_client mocked
- Manual check:
  - Local backend: one generation, one edit using 2 input images, one transparent image, and one enhanced prompt.
  - Space backend: one generation.
  - All of them appear in the gallery.

## Setup

- `uv venv --python 3.12`, then install the dependencies:
  - `torch` (CUDA build)
  - `diffusers` from git
  - `transformers>=5.17`
  - `accelerate`
  - `gradio_client`
  - `fastapi`
  - `uvicorn`
  - `python-multipart`
  - `pillow`
  - `pytest`
  - `httpx`
- `run.ps1` activates the venv and starts `uvicorn app.main:app --host 127.0.0.1 --port 8000`.
- `HF_TOKEN` is optional. It raises the Space quota and allows gated downloads.
