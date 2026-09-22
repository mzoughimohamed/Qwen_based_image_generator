import io
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import config
from app.backends.base import BackendError
from app.main import create_app
from app.storage import Storage
from tests.fakes import FakeBackend


@pytest.fixture
def ctx(tmp_path):
    backends = {"local": FakeBackend("local"), "space": FakeBackend("space")}
    app = create_app(backends=backends, storage=Storage(tmp_path))
    return SimpleNamespace(client=TestClient(app), app=app, backends=backends)


def png(size=(32, 32), mode="RGB"):
    buf = io.BytesIO()
    Image.new(mode, size).save(buf, "PNG")
    return buf.getvalue()


def run_job(ctx, data, files=None):
    r = ctx.client.post("/api/jobs", data=data, files=files)
    assert r.status_code == 200, r.text
    ctx.app.state.jobs.wait_idle()
    return ctx.client.get(f"/api/jobs/{r.json()['job_id']}").json()


def test_health_lists_backends(ctx):
    body = ctx.client.get("/api/health").json()
    assert body["backends"]["local"]["state"] == "ready"
    assert body["backends"]["space"]["state"] == "ready"


def test_options(ctx):
    body = ctx.client.get("/api/options").json()
    assert body["sizes"]["2K"][0] == {"label": "16:9", "width": 2688, "height": 1536}
    assert body["backends"] == ["local", "space"]
    assert body["limits"]["max_images"] == 10


def test_generate_job(ctx):
    job = run_job(ctx, {"backend": "local", "prompt": "a cat", "size": "1024x1024",
                        "seed": "5", "steps": "12", "transparent": "false", "enhance": "false"})
    assert job["status"] == "done"
    meta = job["result"]
    assert (meta["width"], meta["seed"], meta["steps"]) == (1024, 5, 12)
    assert ctx.client.get(f"/outputs/{meta['file']}").status_code == 200
    assert [m["id"] for m in ctx.client.get("/api/history").json()] == [meta["id"]]


def test_edit_with_upload_and_from_result(ctx):
    first = run_job(ctx, {"backend": "local", "prompt": "base"})["result"]
    job = run_job(ctx, {"backend": "space", "prompt": "combine", "from_result": [first["id"]]},
                  files=[("images", ("a.png", png(), "image/png"))])
    assert job["status"] == "done"
    req = ctx.backends["space"].calls[-1]
    assert req.mode == "edit"
    assert len(req.images) == 2
    assert len(job["result"]["input_files"]) == 2


@pytest.mark.parametrize("data", [
    {"backend": "local", "prompt": "   "},
    {"backend": "cloud", "prompt": "p"},
    {"backend": "local", "prompt": "p", "steps": "0"},
    {"backend": "local", "prompt": "p", "size": "huge"},
    {"backend": "local", "prompt": "p", "from_result": ["20260101-000000-abcd"]},
])
def test_rejects_invalid(ctx, data):
    assert ctx.client.post("/api/jobs", data=data).status_code == 400


def test_rejects_too_many_images(ctx):
    files = [("images", (f"{i}.png", png(), "image/png")) for i in range(11)]
    r = ctx.client.post("/api/jobs", data={"backend": "local", "prompt": "p"}, files=files)
    assert r.status_code == 400


def test_rejects_non_image(ctx):
    files = [("images", ("a.txt", b"hello", "text/plain"))]
    r = ctx.client.post("/api/jobs", data={"backend": "local", "prompt": "p"}, files=files)
    assert r.status_code == 400
    assert "not a valid image" in r.json()["detail"]


def test_rejects_oversize(ctx, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 100)
    files = [("images", ("a.png", png((128, 128)), "image/png"))]
    r = ctx.client.post("/api/jobs", data={"backend": "local", "prompt": "p"}, files=files)
    assert r.status_code == 400


def test_backend_not_ready_is_503(ctx):
    ctx.backends["local"].state = "loading"
    r = ctx.client.post("/api/jobs", data={"backend": "local", "prompt": "p"})
    assert r.status_code == 503


def test_job_error_is_reported(ctx):
    ctx.backends["local"].error = BackendError("Out of GPU memory")
    job = run_job(ctx, {"backend": "local", "prompt": "p"})
    assert (job["status"], job["error"]) == ("error", "Out of GPU memory")


def test_delete_history(ctx):
    rid = run_job(ctx, {"backend": "local", "prompt": "p"})["result"]["id"]
    assert ctx.client.delete(f"/api/history/{rid}").status_code == 204
    assert ctx.client.delete(f"/api/history/{rid}").status_code == 404
    assert ctx.client.get("/api/history").json() == []


def test_unknown_job_is_404(ctx):
    assert ctx.client.get("/api/jobs/nope").status_code == 404


def test_index_is_served(ctx):
    r = ctx.client.get("/")
    assert r.status_code == 200
    assert "Qwen-Image" in r.text
