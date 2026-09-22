import threading

from PIL import Image

from app.schemas import JobResult


class FakeBackend:
    def __init__(self, name="local", state="ready", error=None, rewritten=None,
                 warning=None, mode="RGB", gate=None):
        self.name = name
        self.state = state
        self.error = error
        self.rewritten = rewritten
        self.warning = warning
        self.mode = mode
        self.gate = gate
        self.calls = []
        self.started = threading.Event()

    def status(self):
        return {"state": self.state, "detail": f"fake {self.name}"}

    def run(self, req, on_progress):
        self.calls.append(req)
        self.started.set()
        if self.gate is not None:
            self.gate.wait(5)
        if self.error:
            raise self.error
        for step in (1, 2):
            on_progress("generating", step, 2)
        width, height = req.size or (64, 64)
        color = (255, 0, 0, 0) if self.mode == "RGBA" else (255, 0, 0)
        return JobResult(
            image=Image.new(self.mode, (width, height), color),
            seed=req.seed if req.seed is not None else 123,
            rewritten_prompt=self.rewritten,
            warning=self.warning,
        )
