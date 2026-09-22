from datetime import datetime, timedelta

from PIL import Image

from app.schemas import JobRequest, JobResult
from app.storage import Storage


class FakeClock:
    def __init__(self):
        self.now = datetime(2026, 9, 21, 12, 0, 0)

    def __call__(self):
        self.now += timedelta(seconds=1)
        return self.now


def make(tmp_path):
    return Storage(tmp_path / "out", clock=FakeClock())


def result(mode="RGB", size=(64, 32)):
    color = (0, 0, 0, 0) if mode == "RGBA" else (255, 0, 0)
    return JobResult(image=Image.new(mode, size, color), seed=7)


def test_save_writes_png_and_metadata(tmp_path):
    s = make(tmp_path)
    req = JobRequest(backend="local", prompt="a cat", seed=7, steps=30)
    meta = s.save(req, result(), duration_s=1.234)
    assert (s.root / meta["file"]).exists()
    assert meta["mode"] == "generate"
    assert meta["backend"] == "local"
    assert (meta["width"], meta["height"]) == (64, 32)
    assert meta["steps"] == 30
    assert meta["seed"] == 7
    assert meta["duration_s"] == 1.2
    assert meta["input_files"] == []
    assert meta["size_label"] == "custom"
    assert s.get(meta["id"]) == meta


def test_edit_inputs_are_saved(tmp_path):
    s = make(tmp_path)
    req = JobRequest(backend="local", prompt="p", images=[Image.new("RGB", (4, 4))] * 2)
    meta = s.save(req, result(), duration_s=0)
    assert meta["mode"] == "edit"
    assert len(meta["input_files"]) == 2
    for name in meta["input_files"]:
        assert (s.root / name).exists()


def test_rgba_is_preserved(tmp_path):
    s = make(tmp_path)
    meta = s.save(JobRequest(backend="local", prompt="p"), result("RGBA"), duration_s=0)
    with Image.open(s.root / meta["file"]) as im:
        assert im.mode == "RGBA"


def test_space_results_have_no_steps(tmp_path):
    s = make(tmp_path)
    meta = s.save(JobRequest(backend="space", prompt="p"), result(), duration_s=0)
    assert meta["steps"] is None


def test_list_history_newest_first(tmp_path):
    s = make(tmp_path)
    ids = [s.save(JobRequest(backend="local", prompt=str(i)), result(), 0)["id"] for i in range(3)]
    assert [m["id"] for m in s.list_history()] == list(reversed(ids))


def test_list_history_skips_unreadable_files(tmp_path):
    s = make(tmp_path)
    good_id = s.save(JobRequest(backend="local", prompt="ok"), result(), 0)["id"]
    (s.root / "garbage.json").write_text("not json{{{", encoding="utf-8")
    (s.root / "empty.json").write_text("{}", encoding="utf-8")
    assert [m["id"] for m in s.list_history()] == [good_id]


def test_save_leaves_no_tmp_files(tmp_path):
    s = make(tmp_path)
    s.save(JobRequest(backend="local", prompt="p"), result(), 0)
    assert list(s.root.glob("*.tmp")) == []
    assert list(s.root.glob("*.json.tmp")) == []


def test_delete_removes_all_files(tmp_path):
    s = make(tmp_path)
    req = JobRequest(backend="local", prompt="p", images=[Image.new("RGB", (4, 4))])
    meta = s.save(req, result(), 0)
    assert s.delete(meta["id"]) is True
    assert list(s.root.iterdir()) == []
    assert s.delete(meta["id"]) is False


def test_bad_ids_are_rejected(tmp_path):
    s = make(tmp_path)
    assert s.get("../secret") is None
    assert s.delete("..") is False
    assert s.load_image("nope") is None


def test_load_image(tmp_path):
    s = make(tmp_path)
    meta = s.save(JobRequest(backend="local", prompt="p"), result(size=(10, 20)), 0)
    assert s.load_image(meta["id"]).size == (10, 20)
