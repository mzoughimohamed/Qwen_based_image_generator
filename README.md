# Qwen-Image Studio

A local web app for [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1), Qwen's open image generation and editing model. It can:

- **Generate** images from a text prompt, at 1K or 2K presets or a custom size.
- **Edit** images from an instruction, using 1–10 reference images. This includes combining subjects from several images.
- **Make transparent PNGs** (RGBA), including extracting a subject from a photo.
- **Enhance prompts**, rewriting a short prompt into a detailed one with Qwen's prompt-enhancer models.
- **Keep a gallery** of every result with its prompt and settings, so you can reuse settings or edit a result further.

Each job runs on one of two backends, which you choose in the page header:

| Backend | Runs on | Notes |
|---|---|---|
| **Local GPU** (default) | Your NVIDIA GPU, using the open weights via 🤗 Diffusers | Full control over steps, seed and size |
| **HF Space** | The official [Qwen/Qwen-Image-2.1 Space](https://huggingface.co/spaces/Qwen/Qwen-Image-2.1) | No GPU needed. Subject to Hugging Face's usage limits and queue |

The server listens on **127.0.0.1 only** and has no login. It's meant for one person on their own machine.

---

## Requirements

- An NVIDIA GPU. 32 GB of VRAM holds the image model fully on the GPU. Smaller cards fall back to CPU offload automatically, which is slower.
- About **75 GB of free disk** for the models:

  | Model | Size | Used for |
  |---|---|---|
  | `Qwen/Qwen-Image-2.1` | 33.1 GB | Generating and editing |
  | `Qwen/Qwen-Image-2.1-PE-T2I` | 18.8 GB | Enhance prompt (generate) |
  | `Qwen/Qwen-Image-2.1-PE-I2I` | 18.8 GB | Enhance prompt (edit) |

- 64 GB+ of system RAM is recommended if you use prompt enhancement, because the enhancer is kept in RAM between uses.
- Either **Docker Desktop with NVIDIA GPU support**, or **Python 3.12** on Windows.

## Quick start with Docker (recommended)

```powershell
docker compose up -d --build
docker compose logs -f model-download
```

- The first run builds the image (about 12 GB), then the `model-download` service downloads all three models (about 71 GB) into the `hf-cache` Docker volume and exits.
- The `app` service starts only after the download succeeds. Then open **http://127.0.0.1:8000**.
- The Local GPU dot in the header turns green once the model is loaded, which takes about 30–75 seconds.

Everyday commands:

| Task | Command |
|---|---|
| Stop the app (frees the GPU; models stay downloaded) | `docker compose stop app` |
| Start it again | `docker compose start app` |
| Start the app while the enhancers are still downloading | `docker compose up -d --no-deps app` |
| Follow the app logs | `docker compose logs -f app` |
| Rebuild after changing the code | `docker compose build app; docker compose up -d --no-deps app` |
| Stop everything | `docker compose down` |
| Stop everything **and delete the downloaded models** | `docker compose down -v` |

- Later `up` runs skip files that are already downloaded.
- Results are written to `./outputs` on your machine.
- To use a Hugging Face token, put `HF_TOKEN=hf_...` in a `.env` file next to `docker-compose.yml`. A token speeds up downloads and gives you a larger Space quota.

## Running without Docker

```powershell
python -m pip install --user uv
python -m uv venv --python 3.12 .venv
python -m uv pip install --python .venv torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m uv pip install --python .venv -r requirements.txt

.\run.ps1          # http://127.0.0.1:8000
```

- **Model downloads:** the image model downloads on the first start into your Hugging Face cache (`%USERPROFILE%\.cache\huggingface`). Each enhancer downloads the first time you use Enhance prompt.
- **Environment variables:**
  - `$env:QWEN_DISABLE_LOCAL = "1"` skips loading the local model, so only the HF Space backend is used.
  - `$env:HF_TOKEN = "hf_..."` uses your Hugging Face token for downloads and the Space.
  - `$env:QWEN_FORCE_OFFLOAD = "1"` loads the model in CPU-offload mode from the start. See Performance.

## Using the app

- **Generate tab:**
  1. Write a prompt, and optionally a negative prompt.
  2. Pick a size: 2K or 1K presets in 16:9, 9:16, 1:1, 4:3 or 3:4, or a custom size from 256 to 2688 px.
  3. Set the steps (Local only; default 40) and a seed (🎲 picks a random one).
  4. Press **Generate** or <kbd>Ctrl</kbd>+<kbd>Enter</kbd>.
- **Edit tab:**
  1. Drop, browse for or paste up to 10 images. Reorder them with ← →; the numbers match "image 1", "image 2" in your instruction.
  2. Write an instruction, for example *"Change the background to a sunset beach"* or *"Put the cat from image 1 on the sofa in image 2"*.
  3. The size defaults to **Auto**, which matches the first image's aspect ratio.
- **Transparent background:** outputs an RGBA PNG. Combine it with Edit and an instruction like *"Extract the person"* to cut a subject out of a photo.
- **Enhance prompt:** rewrites your prompt first and shows the result under the image. **Use as my prompt** copies it into the prompt box.
  - With the Local backend, the first use loads a 9B model, and from then on the image model runs in CPU-offload mode, which is slower.
  - With the HF Space backend, the enhancement happens on the Space's side.
- **Result actions:**
  - **Download**
  - **Reuse settings**, which fills the form with that image's settings
  - **Edit this image**, which adds it as an input on the Edit tab
  - **Copy seed**
- **Gallery tab:** every result, newest first. You can filter by Generated or Edited, and by Local or Space. Click an image for its details, Reuse, Edit this or Delete.

Jobs run **one at a time** in the order you submit them. The progress bar shows your queue position, the current step or "Enhancing prompt…".

## Where things are stored

- `outputs/<id>.png` is the image.
- `outputs/<id>.json` holds its prompt, rewritten prompt, backend, size, steps, seed, flags and timing.
- `outputs/<id>.input-<n>.png` are the input images of an edit.

Delete an image from the Gallery to remove all of its files.

## HTTP API

The page is built on a small JSON API, which you can also call yourself from the same machine:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Status of each backend: `loading`, `ready`, `error` or `disabled`, with detail |
| `GET` | `/api/options` | Size presets and limits |
| `POST` | `/api/jobs` | Submit a job as multipart form data (fields below). Returns `{"job_id": ...}` |
| `GET` | `/api/jobs/{id}` | Job status, progress and the result metadata when done |
| `GET` | `/api/history` | All saved results, newest first |
| `DELETE` | `/api/history/{id}` | Delete a result |
| `GET` | `/outputs/<file>` | Result and input images |

`POST /api/jobs` accepts these form fields:

| Field | Required | Values |
|---|---|---|
| `backend` | yes | `local` or `space` |
| `prompt` | yes | Text |
| `negative_prompt` | no | Text |
| `size` | no | `auto` or `WIDTHxHEIGHT` |
| `steps` | no | 1–100. Local only |
| `seed` | no | An integer. The Space accepts up to 2147483647 |
| `transparent` | no | `true` or `false` |
| `enhance` | no | `true` or `false` |
| `images` | no | 0–10 image files, up to 20 MB each. Sending any images makes the job an edit |
| `from_result` | no | IDs of gallery results to use as input images |

Example:

```powershell
curl.exe -F backend=local -F "prompt=a lighthouse on a cliff at sunset, oil painting" -F size=2688x1536 http://127.0.0.1:8000/api/jobs
```

## Configuration

Defaults live in `app/config.py`:

| Setting | Default | Meaning |
|---|---|---|
| `MAX_IMAGES` | 10 | Maximum input images per edit |
| `MAX_UPLOAD_BYTES` | 20 MB | Maximum size of each uploaded image |
| `DEFAULT_STEPS` | 40 | Default number of denoising steps |
| `ENHANCER_MAX_NEW_TOKENS` | 4096 | How long the prompt enhancer may think before answering |
| `SPACE_TIMEOUT_S` | 900 | How long to wait for the HF Space before a job fails |
| `SIZE_PRESETS` | 1K / 2K lists | The sizes offered in the UI |

Environment variables:

| Variable | Effect |
|---|---|
| `HF_TOKEN` | Hugging Face token, used for downloads and the Space |
| `QWEN_DISABLE_LOCAL=1` | Don't load the local model; use the HF Space only. Applies to non-Docker runs |
| `QWEN_FORCE_OFFLOAD=1` | Load the model in CPU-offload mode from the start. See Performance |

## Performance

By default the whole model is placed on the GPU. It needs about 31.7 GB, so on a 32 GB card it
fills 98% of the memory and leaves almost nothing for each step's working memory, which then
spills to system RAM and slows everything down.

CPU offload keeps only the part of the model that is currently running on the GPU. On an
RTX 5000 Ada (32 GB) that is far faster, measured with the same prompt, size, steps and seed:

| Mode | 1024x1024, 8 steps | VRAM used |
|---|---|---|
| GPU-resident (default) | 151.6 s | 32.2 GB |
| **CPU offload** | **35.8 s** | 2.8 GB |

With CPU offload, 2688x1536 takes about 6.5 s per step, so a 40-step 2K image takes roughly
4-5 minutes.

To turn it on:

```powershell
# Docker: put this in .env next to docker-compose.yml
QWEN_FORCE_OFFLOAD=1

# Without Docker
$env:QWEN_FORCE_OFFLOAD = "1"; .un.ps1
```

Cards with much more memory than the model needs are likely to be faster without it.

## Troubleshooting

- **The Local GPU dot is red or "unavailable".** Hover over the dot, or open http://127.0.0.1:8000/api/health, to see the exact error.
  - `requires the Torchvision library`: install `torchvision` (see Setup). The Docker image already includes it.
  - `401` or `gated`: run `.\.venv\Scripts\huggingface-cli.exe login`, or set `HF_TOKEN`.
- **"Out of GPU memory".** Use a 1K size or fewer input images. After you use Enhance prompt, the app switches to CPU offload to make room.
- **"HF Space error: Queue is full! Please try again."** The public Space is busy. Try again later or use the Local GPU backend.
- **"HF Space accepts seeds up to 2147483647".** Use a smaller seed. The 🎲 button always picks one that works on both backends.
- **Slow model downloads.** Set `HF_TOKEN`. Unauthenticated downloads are rate-limited.
- **"Invalid host header".** Open the app at `http://127.0.0.1:8000` or `http://localhost:8000`. Other hostnames are rejected on purpose.

## Project layout

```
app/
  main.py             FastAPI app, created by create_app()
  config.py           Model IDs, limits, size presets
  schemas.py          Request and result types, validation, size resolution
  jobs.py             Single-worker FIFO job queue
  storage.py          PNG + JSON results in outputs/
  download_models.py  Pre-fetches all models (Docker model-download service)
  backends/
    local.py          Diffusers QwenImage21Pipeline on the local GPU
    enhancer.py       PE-T2I / PE-I2I prompt enhancers, loaded on demand
    space.py          HF Space backend via gradio_client
static/               index.html, app.js, style.css (no build step)
tests/                pytest suite; no GPU or network needed
docs/superpowers/     Design spec and implementation plan
Dockerfile, docker-compose.yml, run.ps1
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

All tests replace the model, the GPU and the network with fakes, so the whole suite runs in a few seconds.
