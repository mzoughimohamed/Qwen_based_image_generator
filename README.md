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
python -m uv pip install --python .venv torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m uv pip install --python .venv -r requirements.txt
```

## Run

```powershell
.\run.ps1          # http://127.0.0.1:8000
```

- The first local run downloads about 14 GB of weights. The prompt enhancers are about 18 GB each and download on first use.
- `$env:QWEN_DISABLE_LOCAL = "1"` runs with the HF Space backend only.
- `$env:HF_TOKEN = "hf_..."` gives a larger Space quota and access to gated downloads.
- If the model download fails with 401/gated, run `.\.venv\Scripts\huggingface-cli.exe login` (or set `HF_TOKEN`).
- If the model doesn't fit on your GPU, it falls back to CPU offload automatically (slower); the Local GPU dot's tooltip shows which.

Results are saved in `outputs/`, one PNG and one JSON file per image.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Docker

This setup needs Docker Desktop with NVIDIA GPU support. Build and start everything with:

```powershell
docker compose up -d --build
docker compose logs -f model-download   # first run: downloads ~50 GB (image model + both enhancers)
```

- The `model-download` service pre-fetches `Qwen/Qwen-Image-2.1`, `Qwen/Qwen-Image-2.1-PE-T2I` and `Qwen/Qwen-Image-2.1-PE-I2I` into the `hf-cache` volume, then exits.
- `app` starts only after the download succeeds. It then serves http://127.0.0.1:8000.
- Later `up` runs skip files that are already downloaded.
- Results appear in `./outputs`.
- To pass a Hugging Face token, set `HF_TOKEN` in your shell or in a `.env` file next to `docker-compose.yml`.
- `docker compose down` stops the stack and keeps the downloaded models. `docker compose down -v` also deletes them.
