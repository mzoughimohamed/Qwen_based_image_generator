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
