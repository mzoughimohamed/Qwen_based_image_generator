import threading

import pytest

from app.backends.base import BackendError
from app.jobs import JobQueue
from app.schemas import ValidationError, build_request
from app.storage import Storage
from tests.fakes import FakeBackend


def make_queue(tmp_path, **backends):
    return JobQueue(backends, Storage(tmp_path))


def test_job_completes_and_saves(tmp_path):
    q = make_queue(tmp_path, local=FakeBackend("local"))
    job = q.submit(build_request(backend="local", prompt="a cat"))
    q.wait_idle()
    snap = q.snapshot(job.id)
    assert snap["status"] == "done"
    assert snap["result_id"] == snap["result"]["id"]
    assert snap["result"]["prompt"] == "a cat"
    assert (snap["phase"], snap["step"], snap["total_steps"]) == ("generating", 2, 2)
    assert snap["queue_position"] is None


def test_fifo_order_and_queue_positions(tmp_path):
    gate = threading.Event()
    fake = FakeBackend("local", gate=gate)
    q = make_queue(tmp_path, local=fake)
    jobs = [q.submit(build_request(backend="local", prompt=str(i))) for i in range(3)]
    assert fake.started.wait(5)
    assert q.snapshot(jobs[0].id)["status"] == "running"
    assert q.snapshot(jobs[1].id)["queue_position"] == 1
    assert q.snapshot(jobs[2].id)["queue_position"] == 2
    gate.set()
    q.wait_idle()
    assert [r.prompt for r in fake.calls] == ["0", "1", "2"]


def test_backend_error_message(tmp_path):
    q = make_queue(tmp_path, local=FakeBackend("local", error=BackendError("boom")))
    job = q.submit(build_request(backend="local", prompt="p"))
    q.wait_idle()
    snap = q.snapshot(job.id)
    assert (snap["status"], snap["error"]) == ("error", "boom")


def test_unexpected_exception_is_reported(tmp_path):
    q = make_queue(tmp_path, local=FakeBackend("local", error=RuntimeError("bad")))
    job = q.submit(build_request(backend="local", prompt="p"))
    q.wait_idle()
    assert q.snapshot(job.id)["error"] == "RuntimeError: bad"


def test_rewritten_prompt_and_warning_propagate(tmp_path):
    fake = FakeBackend("local", rewritten="rich", warning="careful")
    q = make_queue(tmp_path, local=fake)
    job = q.submit(build_request(backend="local", prompt="p", enhance=True))
    q.wait_idle()
    snap = q.snapshot(job.id)
    assert (snap["rewritten_prompt"], snap["warning"]) == ("rich", "careful")
    assert snap["result"]["rewritten_prompt"] == "rich"


def test_unknown_backend_rejected(tmp_path):
    q = make_queue(tmp_path, local=FakeBackend("local"))
    with pytest.raises(ValidationError):
        q.submit(build_request(backend="space", prompt="p"))


def test_unknown_job_snapshot_is_none(tmp_path):
    assert make_queue(tmp_path, local=FakeBackend()).snapshot("missing") is None


def test_snapshot_consistency_while_running(tmp_path):
    gate = threading.Event()
    fake = FakeBackend("local", gate=gate)
    q = make_queue(tmp_path, local=fake)
    job = q.submit(build_request(backend="local", prompt="test"))
    assert fake.started.wait(5)
    # Poll snapshot while running; must not observe torn update
    snap = q.snapshot(job.id)
    assert snap["status"] == "running"
    gate.set()
    q.wait_idle()
    snap = q.snapshot(job.id)
    assert snap["status"] == "done"
