"""Pre-fetch every model into the Hugging Face cache (Docker `model-download` service)."""

from __future__ import annotations

import os

from huggingface_hub import snapshot_download

from app import config

MODEL_IDS = (config.MODEL_ID, config.PE_T2I_ID, config.PE_I2I_ID)


def download_all(download=None, token: str | None = None) -> list[str]:
    download = download or snapshot_download
    paths = []
    for model_id in MODEL_IDS:
        print(f"Downloading {model_id} …", flush=True)
        paths.append(download(repo_id=model_id, token=token))
        print(f"Ready: {model_id}", flush=True)
    return paths


def main() -> None:
    download_all(token=os.environ.get("HF_TOKEN") or None)


if __name__ == "__main__":
    main()
