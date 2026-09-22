from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from PIL import Image, ImageOps

from app import config

BACKENDS = ("local", "space")

_SIZE_RE = re.compile(r"^(\d+)x(\d+)$")
_RATIO_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*$")


class ValidationError(ValueError):
    pass


@dataclass
class JobRequest:
    backend: str
    prompt: str
    negative_prompt: str = " "
    images: list[Image.Image] = field(default_factory=list)
    size: tuple[int, int] | None = None
    steps: int = config.DEFAULT_STEPS
    seed: int | None = None
    transparent: bool = False
    enhance: bool = False

    @property
    def mode(self) -> str:
        return "edit" if self.images else "generate"


@dataclass
class JobResult:
    image: Image.Image
    seed: int
    rewritten_prompt: str | None = None
    warning: str | None = None


def _round16(value: int) -> int:
    return int(value / 16 + 0.5) * 16


def parse_size(value: str | None) -> tuple[int, int] | None:
    if value is None or value.strip().lower() in ("", "auto"):
        return None
    match = _SIZE_RE.match(value.strip().lower())
    if not match:
        raise ValidationError(f"Invalid size '{value}' (use 'auto' or WIDTHxHEIGHT)")
    width, height = _round16(int(match.group(1))), _round16(int(match.group(2)))
    for dim in (width, height):
        if not config.CUSTOM_MIN <= dim <= config.CUSTOM_MAX:
            raise ValidationError(
                f"Size must be between {config.CUSTOM_MIN} and {config.CUSTOM_MAX} pixels per side"
            )
    return width, height


def parse_ratio(text: str | None) -> float | None:
    match = _RATIO_RE.match(text or "")
    if not match or float(match.group(2)) == 0:
        return None
    return float(match.group(1)) / float(match.group(2))


def nearest_preset(ratio: float, tier: str = "2K") -> tuple[int, int]:
    _, width, height = min(
        config.SIZE_PRESETS[tier], key=lambda p: abs(math.log(p[1] / p[2]) - math.log(ratio))
    )
    return width, height


def resolve_local_size(size, images, ratio_hint: str | None = None) -> tuple[int, int]:
    if size:
        return size
    ratio = parse_ratio(ratio_hint)
    if ratio:
        return nearest_preset(ratio)
    if images:
        width, height = images[0].size
        return nearest_preset(width / height)
    return config.DEFAULT_T2I_SIZE


def size_label(width: int, height: int) -> str:
    for tier, presets in config.SIZE_PRESETS.items():
        for label, w, h in presets:
            if (w, h) == (width, height):
                return f"{tier} {label}"
    return "custom"


def normalize_image(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    has_alpha = img.mode in ("RGBA", "LA", "PA") or (
        img.mode == "P" and "transparency" in img.info
    )
    return img.convert("RGBA" if has_alpha else "RGB")


def build_request(
    *,
    backend: str,
    prompt: str,
    negative_prompt: str | None = None,
    images=(),
    size: str | None = None,
    steps: int | None = None,
    seed: int | None = None,
    transparent: bool = False,
    enhance: bool = False,
) -> JobRequest:
    if backend not in BACKENDS:
        raise ValidationError(f"Unknown backend '{backend}'")
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValidationError("Prompt must not be empty")
    images = list(images)
    if len(images) > config.MAX_IMAGES:
        raise ValidationError(f"At most {config.MAX_IMAGES} input images are allowed")
    steps = config.DEFAULT_STEPS if steps is None else int(steps)
    if not config.MIN_STEPS <= steps <= config.MAX_STEPS:
        raise ValidationError(f"Steps must be between {config.MIN_STEPS} and {config.MAX_STEPS}")
    if seed is not None and not 0 <= seed <= config.MAX_SEED:
        raise ValidationError(f"Seed must be between 0 and {config.MAX_SEED}")
    return JobRequest(
        backend=backend,
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt and negative_prompt.strip() else " ",
        images=images,
        size=parse_size(size),
        steps=steps,
        seed=seed,
        transparent=bool(transparent),
        enhance=bool(enhance),
    )
