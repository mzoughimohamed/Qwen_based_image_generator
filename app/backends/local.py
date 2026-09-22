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
