from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from app import config

_DECODER = json.JSONDecoder()


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_enhancer_output(text: str) -> tuple[str, str | None]:
    tail = text.rsplit("</think>", 1)[-1]
    found = None
    for i, ch in enumerate(tail):
        if ch != "{":
            continue
        try:
            obj, _ = _DECODER.raw_decode(tail, i)
        except json.JSONDecodeError:
            continue
        prompt = obj.get("rewritten_prompt") if isinstance(obj, dict) else None
        if isinstance(prompt, str) and prompt.strip():
            found = obj
    if found is None:
        raise ValueError("enhancer output had no rewritten_prompt JSON")
    ratio = found.get("wh_ratio")
    return found["rewritten_prompt"].strip(), ratio if isinstance(ratio, str) and ratio else None


@dataclass
class LoadedModel:
    model: Any
    processor: Any
    system_prompt: str | None
    kind: str  # "t2i" | "i2i"


def _system_prompt(model_id: str) -> str | None:
    import huggingface_hub

    try:
        path = huggingface_hub.hf_hub_download(model_id, "system_prompt.txt")
    except Exception:  # noqa: BLE001 - repo may not ship one
        return None
    return Path(path).read_text(encoding="utf-8").strip()


def load_pe_model(model_id: str) -> LoadedModel:
    if model_id == config.PE_I2I_ID:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16).eval()
        processor = AutoProcessor.from_pretrained(model_id)
        kind = "i2i"
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).eval()
        processor = AutoTokenizer.from_pretrained(model_id)
        kind = "t2i"
    return LoadedModel(model, processor, _system_prompt(model_id), kind)


def run_pe_model(lm: LoadedModel, prompt: str, images, max_new_tokens: int, device: str) -> str:
    if lm.kind == "i2i":
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        if lm.system_prompt:
            messages.insert(0, {"role": "system",
                                "content": [{"type": "text", "text": lm.system_prompt}]})
        inputs = lm.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", enable_thinking=True,
        ).to(device)
        tokenizer = lm.processor.tokenizer
    else:
        messages = [{"role": "user", "content": prompt}]
        if lm.system_prompt:
            messages.insert(0, {"role": "system", "content": lm.system_prompt})
        text = lm.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        inputs = lm.processor(text, return_tensors="pt").to(device)
        tokenizer = lm.processor
    with torch.no_grad():
        out = lm.model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=1.0, top_p=0.95, top_k=20,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)


class PromptEnhancer:
    def __init__(self, load_fn=load_pe_model, run_fn=run_pe_model, device: str = "cuda",
                 max_new_tokens: int = config.ENHANCER_MAX_NEW_TOKENS):
        self._load_fn = load_fn
        self._run_fn = run_fn
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._resident: LoadedModel | None = None
        self._resident_id: str | None = None

    @property
    def resident_id(self) -> str | None:
        return self._resident_id

    def enhance(self, prompt: str, images) -> tuple[str, str | None]:
        model_id = config.PE_I2I_ID if images else config.PE_T2I_ID
        lm = self._ensure(model_id)
        try:
            lm.model.to(self._device)
            text = self._run_fn(lm, prompt, images, self._max_new_tokens, self._device)
        except Exception:
            self.unload()
            raise
        lm.model.to("cpu")
        free_cuda()
        return parse_enhancer_output(text)

    def unload(self) -> None:
        if self._resident is not None:
            try:
                self._resident.model.to("cpu")
            except Exception:  # noqa: BLE001 - best effort while recovering
                pass
        self._resident = None
        self._resident_id = None
        free_cuda()

    def _ensure(self, model_id: str) -> LoadedModel:
        if self._resident_id != model_id:
            self.unload()
            self._resident = self._load_fn(model_id)
            self._resident_id = model_id
        return self._resident
