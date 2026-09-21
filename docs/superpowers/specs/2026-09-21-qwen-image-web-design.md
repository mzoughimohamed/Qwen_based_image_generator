# Qwen-Image Web — Design

Date: 2026-09-21
Status: Approved in brainstorming, pending spec review

## Goal

A small local website that serves **Qwen-Image-2.1** (`Qwen/Qwen-Image-2.1`, Diffusers
`QwenImage21Pipeline`) on the user's own GPU (RTX 5000 Ada, 32 GB) for a single user at
`http://127.0.0.1`. It supports:

1. Text-to-image generation
2. Image editing (uploaded image + instruction)
3. Transparent (RGBA) output, including subject extraction
4. A history gallery of all results

Out of scope: multi-user access, authentication, cloud APIs, LoRAs, ComfyUI, prompt enhancers.

## Architecture

FastAPI backend + Diffusers, static HTML/CSS/JS frontend served by the same process.
No Node, no build step, no database.

```
image_generator/
  .venv/                 # Python 3.12 via uv (PyTorch wheels may not exist for 3.14)
  app/
    main.py              # FastAPI app: API routes, serves static/ and outputs/
    model.py             # pipeline loading + generate()/edit()
    jobs.py              # single-worker job queue
    storage.py           # PNG + JSON metadata persistence, history listing
    config.py            # constants: aspect ratios, presets, paths, limits
  static/
    index.html, app.js, style.css
  outputs/               # results: <id>.png + <id>.json (+ <id>.input.png for edits)
  tests/
  requirements.txt
  run.ps1
```

### Model (`app/model.py`)

- Loads `Qwen/Qwen-Image-2.1` in `torch.bfloat16` in a background thread at server start.
- Tries `.to("cuda")`; on `torch.OutOfMemoryError` reloads with `enable_model_cpu_offload()`.
- Exposes a `ModelBackend` interface so tests can inject a fake:
  - `generate(prompt, negative_prompt, width, height, steps, seed, transparent, on_step) -> PIL.Image`
  - `edit(image, prompt, steps, seed, transparent, on_step) -> PIL.Image`
- `transparent=True` appends the model card's RGBA phrase to the prompt:
  `"This is an RGBA image with transparency. The image has alpha channel and the background is transparent."`
  The output is saved as a PNG with alpha (mode `RGBA`).
- Subject extraction = `edit` with `transparent=True` and an instruction such as "extract the person".
- Progress reported through the Diffusers `callback_on_step_end` hook → `on_step(step, total)`.
- `negative_prompt` is passed only if the pipeline signature accepts it (checked via `inspect.signature`).

### Resolutions and presets (`app/config.py`)

Supported aspect ratios (Quality preset, from the model card):

| Ratio | Size |
|---|---|
| 1:1 | 2048×2048 |
| 4:3 | 2400×1792 |
| 3:4 | 1792×2400 |
| 3:2 | 2528×1696 |
| 2:3 | 1696×2528 |
| 16:9 | 2752×1536 |
| 9:16 | 1536×2752 |

- **Quality**: sizes above, 40 steps default.
- **Fast**: half each dimension (rounded down to a multiple of 16), 20 steps default.
- Steps range 1–100. Seed: integer 0–2³²−1; random if omitted, always recorded.

### Jobs (`app/jobs.py`)

- One worker thread consumes a FIFO queue, so only one GPU job runs at a time.
- Job states: `queued → running → done | error`. Fields: `id, status, step, total_steps,
  queue_position, result_id, error`.
- Jobs are kept in memory (lost on restart; the results themselves are saved to disk).

### Storage (`app/storage.py`)

- Result id: `YYYYMMDD-HHMMSS-<4 hex>`.
- Files: `<id>.png`, `<id>.json`; for edits also `<id>.input.png`.
- Metadata JSON: `id, mode ("generate"|"edit"), prompt, negative_prompt, width, height,
  aspect, preset, steps, seed, transparent, duration_s, created_at, input_file`.
- `list_history()` returns metadata sorted newest-first; `delete(id)` removes all files for the id.

### API (`app/main.py`)

| Method | Path | Body / Response |
|---|---|---|
| GET | `/api/health` | `{status: "loading"|"ready"|"error", offload: bool, error?}` |
| GET | `/api/options` | aspect ratios, presets, limits |
| POST | `/api/generate` | JSON `{prompt, negative_prompt?, aspect, preset, steps?, seed?, transparent}` → `{job_id}` |
| POST | `/api/edit` | multipart `image` file + form fields `prompt, steps?, seed?, transparent` → `{job_id}` |
| GET | `/api/jobs/{id}` | job state |
| GET | `/api/history` | list of metadata |
| DELETE | `/api/history/{id}` | 204 |
| GET | `/outputs/...` | static result files |
| GET | `/` | `static/index.html` |

The server binds to `127.0.0.1` only. There is no authentication.

## Frontend

A single page with three tabs and a dark theme, written in plain HTML/CSS/JS.

- **Generate tab:**
  - prompt textarea, with Ctrl+Enter to submit
  - collapsible negative prompt
  - aspect-ratio buttons and a Quality/Fast toggle
  - Transparent toggle
  - steps field
  - seed field with a random (🎲) button
  - Generate button
- **Edit tab:**
  - drag-drop or file picker for the input image, with a preview
  - instruction textarea
  - steps, seed and Transparent controls
  - Edit button
- **Result panel** (shared by Generate and Edit):
  - progress bar ("Queued (#2)" / "Step 12/40")
  - the result image, on a checkerboard background when transparent
  - a before/after view for edits
  - actions: Download, Reuse settings, Edit this image, Copy seed
- **Gallery tab:**
  - thumbnail grid, newest first
  - filter: All / Generated / Edited
  - clicking a thumbnail opens a detail view with the full prompt and settings, plus Reuse, Edit this and Delete
- While `/api/health` reports `loading`, a banner says "Loading model…" and the submit buttons are disabled.
- The frontend polls `/api/jobs/{id}` every 1 s.

## Error handling

- **Model not ready:** submit endpoints return 503. The UI disables its buttons until the model is ready.
- **CUDA OOM during a job:** the job ends in `error` with a readable message, `torch.cuda.empty_cache()` runs, and the server stays up.
- **Invalid input:** the server returns 400 for any of these:
  - an empty or whitespace-only prompt
  - an unknown aspect or preset
  - steps out of range
  - an upload that is not decodable by PIL
  - an upload larger than 20 MB
- **Edit input size:** the image is converted to RGB(A) and resized to the supported resolution whose aspect ratio is closest to the input's.
- **Model download:** the first run downloads the weights from Hugging Face. If access is gated, the user runs `huggingface-cli login`.

## Testing

- pytest with a `FakeBackend` (returns a solid-color image and calls `on_step`), so no GPU is needed. Tests cover:
  - input validation for each endpoint
  - the job lifecycle and FIFO ordering
  - saving and listing in storage, the metadata round-trip, and delete
  - that a transparent result is saved as an RGBA PNG
  - upload validation (non-image, oversize) and the resize to the nearest ratio
- Manual check on the real GPU: one generation, one edit, one transparent image, then confirm all three appear in the gallery.

## Setup

- `uv venv --python 3.12`, then install the dependencies:
  - `torch` (CUDA build)
  - `diffusers` from git
  - `transformers>=5.17`
  - `accelerate`
  - `fastapi`
  - `uvicorn`
  - `python-multipart`
  - `pillow`
  - `pytest`
  - `httpx`
- `run.ps1` activates the venv and starts `uvicorn app.main:app --host 127.0.0.1 --port 8000`.
