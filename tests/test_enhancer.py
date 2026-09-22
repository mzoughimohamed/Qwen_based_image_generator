import pytest
from PIL import Image

from app import config
from app.backends.enhancer import LoadedModel, PromptEnhancer, parse_enhancer_output


def test_parse_after_think_block():
    text = ('<think>draft {"rewritten_prompt": "draft"}</think>\n'
            '{"rewritten_prompt": "A red fox", "wh_ratio": "16:9"}<|im_end|>')
    assert parse_enhancer_output(text) == ("A red fox", "16:9")


def test_parse_without_think_and_empty_ratio():
    assert parse_enhancer_output('{"rewritten_prompt": "x", "wh_ratio": ""}') == ("x", None)


def test_parse_i2i_extra_keys():
    text = '</think>{"rewritten_prompt": "y", "wh_ratio": "", "ratio_follow": "<image1>"}'
    assert parse_enhancer_output(text) == ("y", None)


def test_parse_takes_last_object():
    text = '{"rewritten_prompt": "a"} and then {"rewritten_prompt": "b", "wh_ratio": "1:1"}'
    assert parse_enhancer_output(text) == ("b", "1:1")


@pytest.mark.parametrize("text", ["no json here", '{"rewritten_prompt": ""}', '{"other": 1}'])
def test_parse_malformed_raises(text):
    with pytest.raises(ValueError):
        parse_enhancer_output(text)


class FakeModel:
    def __init__(self):
        self.devices = []

    def to(self, device):
        self.devices.append(device)
        return self


GOOD = '{"rewritten_prompt": "rich", "wh_ratio": "9:16"}'


def make(outputs):
    loads, models = [], {}

    def load_fn(model_id):
        loads.append(model_id)
        models[model_id] = FakeModel()
        kind = "i2i" if model_id == config.PE_I2I_ID else "t2i"
        return LoadedModel(models[model_id], None, "sys", kind)

    def run_fn(lm, prompt, images, max_new_tokens, device):
        out = outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    return PromptEnhancer(load_fn=load_fn, run_fn=run_fn, device="gpu"), loads, models


def test_lazy_load_and_reuse():
    enh, loads, models = make([GOOD, GOOD])
    assert enh.resident_id is None
    assert enh.enhance("p", []) == ("rich", "9:16")
    assert enh.enhance("p", []) == ("rich", "9:16")
    assert loads == [config.PE_T2I_ID]
    assert models[config.PE_T2I_ID].devices == ["gpu", "cpu", "gpu", "cpu"]


def test_switching_mode_unloads_other():
    enh, loads, models = make([GOOD, GOOD])
    enh.enhance("p", [])
    enh.enhance("p", [Image.new("RGB", (4, 4))])
    assert loads == [config.PE_T2I_ID, config.PE_I2I_ID]
    assert enh.resident_id == config.PE_I2I_ID
    assert models[config.PE_T2I_ID].devices[-1] == "cpu"


def test_generation_error_unloads():
    enh, _, _ = make([RuntimeError("CUDA out of memory")])
    with pytest.raises(RuntimeError):
        enh.enhance("p", [])
    assert enh.resident_id is None


def test_unparseable_output_keeps_model():
    enh, _, models = make(["garbage"])
    with pytest.raises(ValueError):
        enh.enhance("p", [])
    assert enh.resident_id == config.PE_T2I_ID
    assert models[config.PE_T2I_ID].devices[-1] == "cpu"
