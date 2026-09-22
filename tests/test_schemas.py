import pytest
from PIL import Image

from app import config
from app.schemas import (
    ValidationError,
    build_request,
    nearest_preset,
    normalize_image,
    parse_ratio,
    parse_size,
    resolve_local_size,
    size_label,
)


def img(w=8, h=8, mode="RGB"):
    return Image.new(mode, (w, h))


@pytest.mark.parametrize("value", [None, "", "auto", "AUTO", " auto "])
def test_parse_size_auto(value):
    assert parse_size(value) is None


def test_parse_size_rounds_to_multiple_of_16():
    assert parse_size("1000x1010") == (1008, 1008)
    assert parse_size("2048x2048") == (2048, 2048)


@pytest.mark.parametrize("value", ["100x2048", "4000x1024", "big", "12x", "x12"])
def test_parse_size_rejects_invalid(value):
    with pytest.raises(ValidationError):
        parse_size(value)


def test_parse_ratio():
    assert parse_ratio("16:9") == pytest.approx(16 / 9)
    assert parse_ratio("") is None
    assert parse_ratio(None) is None
    assert parse_ratio("abc") is None
    assert parse_ratio("1:0") is None


def test_nearest_preset():
    assert nearest_preset(1.7) == (2688, 1536)
    assert nearest_preset(0.75) == (1728, 2368)
    assert nearest_preset(1.0) == (2048, 2048)
    assert nearest_preset(1.0, tier="1K") == (1024, 1024)


def test_resolve_local_size_explicit_wins():
    assert resolve_local_size((1024, 1024), [img(300, 100)], "9:16") == (1024, 1024)


def test_resolve_local_size_ratio_hint_beats_image():
    assert resolve_local_size(None, [img(300, 100)], "9:16") == (1536, 2688)


def test_resolve_local_size_from_first_image():
    assert resolve_local_size(None, [img(400, 300), img(100, 300)]) == (2368, 1728)


def test_resolve_local_size_default():
    assert resolve_local_size(None, [], None) == config.DEFAULT_T2I_SIZE


def test_size_label():
    assert size_label(1344, 768) == "1K 16:9"
    assert size_label(2048, 2048) == "2K 1:1"
    assert size_label(1000, 1000) == "custom"


def test_build_request_defaults():
    req = build_request(backend="local", prompt="  a cat  ")
    assert req.prompt == "a cat"
    assert req.negative_prompt == " "
    assert req.steps == config.DEFAULT_STEPS
    assert req.seed is None
    assert req.size is None
    assert req.mode == "generate"


def test_build_request_edit_mode_and_size():
    req = build_request(backend="space", prompt="p", images=[img()], size="1024x1024",
                        negative_prompt="blurry", seed=0, steps=12, transparent=True, enhance=True)
    assert req.mode == "edit"
    assert req.size == (1024, 1024)
    assert req.negative_prompt == "blurry"
    assert (req.seed, req.steps, req.transparent, req.enhance) == (0, 12, True, True)


@pytest.mark.parametrize("kwargs", [
    {"backend": "cloud"},
    {"prompt": "   "},
    {"steps": 0},
    {"steps": 101},
    {"seed": -1},
    {"seed": 2**32},
    {"images": [Image.new("RGB", (8, 8)) for _ in range(11)]},
    {"size": "big"},
])
def test_build_request_rejects(kwargs):
    base = {"backend": "local", "prompt": "a cat"}
    base.update(kwargs)
    with pytest.raises(ValidationError):
        build_request(**base)


def test_normalize_image_modes():
    assert normalize_image(img(mode="RGBA")).mode == "RGBA"
    assert normalize_image(img(mode="LA")).mode == "RGBA"
    assert normalize_image(img(mode="L")).mode == "RGB"
    assert normalize_image(img(mode="CMYK")).mode == "RGB"
