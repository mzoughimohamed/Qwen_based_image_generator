from types import SimpleNamespace

import pytest
from PIL import Image

from app import config
from app.backends import space
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


class NeverDoneJob:
    """A job that never completes, to exercise the timeout path."""

    def __init__(self):
        self.cancelled = False

    def done(self):
        return False

    def status(self):
        return SimpleNamespace(rank=None)

    def cancel(self):
        self.cancelled = True

    def result(self):  # pragma: no cover - never reached
        raise AssertionError("result() should not be called after a timeout")


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


def test_non_enhanced_job_ignores_echoed_prompt(tmp_path):
    # The Space may echo the original prompt back in the "rewritten" slot even
    # when enhance was never requested; that must not surface as a rewrite.
    client = FakeClient(FakeJob((output_png(tmp_path), 1, "a cat")))
    result = make(client).run(req(enhance=False), lambda *e: None)
    assert result.rewritten_prompt is None


def test_quota_error_is_friendly():
    client = FakeClient(submit_error=Exception("You have exceeded your GPU quota (60s left)"))
    with pytest.raises(BackendError, match="quota"):
        make(client).run(req(), lambda *e: None)


def test_friendly_error_generic():
    assert friendly_space_error(Exception("boom")) == "HF Space error: boom"


def test_seed_over_space_max_is_rejected_before_network(tmp_path):
    client = FakeClient(FakeJob((output_png(tmp_path), 2**31, "")))
    with pytest.raises(BackendError, match="seeds up to"):
        make(client).run(req(seed=2**31), lambda *e: None)
    assert client.calls == []


def test_seed_at_space_max_is_accepted(tmp_path):
    client = FakeClient(FakeJob((output_png(tmp_path), 2**31 - 1, "")))
    result = make(client).run(req(seed=2**31 - 1), lambda *e: None)
    assert client.calls
    assert result.seed == 2**31 - 1


def test_reconnects_after_submit_failure(tmp_path):
    good_client = FakeClient(FakeJob((output_png(tmp_path), 1, "")))
    clients = [FakeClient(submit_error=Exception("boom")), good_client]
    factory = mock_factory(clients)

    backend = SpaceBackend(client_factory=factory, poll_interval=0)
    with pytest.raises(BackendError):
        backend.run(req(), lambda *e: None)

    backend.run(req(), lambda *e: None)
    assert factory.calls == 2


def mock_factory(clients):
    def factory():
        factory.calls += 1
        return clients[factory.calls - 1]

    factory.calls = 0
    return factory


def test_timeout_cancels_job_and_raises_backend_error():
    job = NeverDoneJob()
    client = FakeClient(job)
    backend = SpaceBackend(client_factory=lambda: client, poll_interval=0, timeout_s=0.05)
    with pytest.raises(BackendError, match="did not respond"):
        backend.run(req(), lambda *e: None)
    assert job.cancelled is True


def test_default_client_passes_token_not_hf_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    captured = {}

    class FakeGradioClient:
        def __init__(self, src, **kwargs):
            captured["src"] = src
            captured.update(kwargs)

    monkeypatch.setattr("gradio_client.Client", FakeGradioClient)

    space._default_client()

    assert captured["src"] == config.SPACE_ID
    assert captured["token"] == "secret-token"
    assert "hf_token" not in captured
