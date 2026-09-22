from __future__ import annotations

import json
import os
import re
import secrets
from datetime import datetime
from pathlib import Path

from PIL import Image

from app.schemas import JobRequest, JobResult, size_label

_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")


def _savable(img: Image.Image) -> Image.Image:
    if img.mode in ("RGB", "RGBA"):
        return img
    has_alpha = "A" in img.mode or "transparency" in img.info
    return img.convert("RGBA" if has_alpha else "RGB")


class Storage:
    def __init__(self, root: Path, clock=datetime.now):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._clock = clock

    def new_id(self, now: datetime | None = None) -> str:
        now = now or self._clock()
        return f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"

    def save(self, req: JobRequest, result: JobResult, duration_s: float) -> dict:
        now = self._clock()
        rid = self.new_id(now)
        while (self.root / f"{rid}.json").exists():
            rid = self.new_id(now)

        input_files = []
        for i, img in enumerate(req.images):
            name = f"{rid}.input-{i}.png"
            _savable(img).save(self.root / name)
            input_files.append(name)

        image = _savable(result.image)
        image.save(self.root / f"{rid}.png")
        width, height = image.size
        meta = {
            "id": rid,
            "mode": req.mode,
            "backend": req.backend,
            "prompt": req.prompt,
            "rewritten_prompt": result.rewritten_prompt,
            "negative_prompt": req.negative_prompt,
            "width": width,
            "height": height,
            "size_label": size_label(width, height),
            "steps": req.steps if req.backend == "local" else None,
            "seed": result.seed,
            "transparent": req.transparent,
            "enhance": req.enhance,
            "warning": result.warning,
            "file": f"{rid}.png",
            "input_files": input_files,
            "duration_s": round(duration_s, 1),
            "created_at": now.isoformat(timespec="microseconds"),
        }
        final_path = self.root / f"{rid}.json"
        tmp_path = self.root / f"{rid}.json.tmp"
        tmp_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        os.replace(tmp_path, final_path)
        return meta

    def get(self, rid: str) -> dict | None:
        if not _ID_RE.match(rid or ""):
            return None
        path = self.root / f"{rid}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_history(self) -> list[dict]:
        metas = []
        for p in self.root.glob("*.json"):
            try:
                meta = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(meta, dict) or "id" not in meta or "created_at" not in meta:
                continue
            metas.append(meta)
        return sorted(metas, key=lambda m: (m["created_at"], m["id"]), reverse=True)

    def load_image(self, rid: str) -> Image.Image | None:
        meta = self.get(rid)
        if meta is None:
            return None
        with Image.open(self.root / meta["file"]) as im:
            im.load()
            return im.copy()

    def delete(self, rid: str) -> bool:
        meta = self.get(rid)
        if meta is None:
            return False
        for name in [meta["file"], f"{rid}.json", *meta["input_files"]]:
            (self.root / name).unlink(missing_ok=True)
        return True
