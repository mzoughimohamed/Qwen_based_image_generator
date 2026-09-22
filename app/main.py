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
    import os

    from app.backends.local import LocalBackend
    from app.backends.space import SpaceBackend

    local = LocalBackend(enabled=os.environ.get("QWEN_DISABLE_LOCAL") != "1")
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
