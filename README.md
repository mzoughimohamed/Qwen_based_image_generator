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
