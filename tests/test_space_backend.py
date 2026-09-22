from types import SimpleNamespace

import pytest
from PIL import Image

from app import config
from app.backends.base import BackendError
from app.backends.space import SPACE_API_NAME, SpaceBackend, build_space_args, friendly_space_error
from app.schemas import build_request

# The real Space's "Log directory" textbox defaults to this value (verified via
# view_api()); we pass it through unchanged rather than the brief's assumed "".
_LOG_DIR = "./generation_logs_paper_case"


def req(**kwargs):
    kwargs.setdefault("prompt", "p")
    return build_request(backend="space", **kwargs)


def test_args_auto_size_random_seed():
    assert build_space_args(req(), "p", None) == [None, "p", False, False, _LOG_DIR, 0, True, 1536, 2688, " "]


def test_args_custom_size_fixed_seed_enhance():
    args = build_space_args(req(size="1024x768", seed=9, enhance=True, negative_prompt="ugly"), "p", None)
    assert args == [None, "p", True, True, _LOG_DIR, 9, False, 768, 1024, "ugly"]


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
