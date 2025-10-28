"""
LLM based Valence-Arousal regression with instruct-style prompts.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftConfig, PeftModel

from .va_prompts import build_instruct_prompt

LOGGER = logging.getLogger(__name__)


def clamp(value: float, low: float = 1.0, high: float = 9.0) -> float:
    return max(low, min(high, value))


@dataclass
class VAPredictorConfig:
    model_name_or_path: str
    device: Optional[str] = None
    max_new_tokens: int = 7
    temperature: float = 1
    top_p: float = 0.95
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    cache_dir: Optional[str] = None


class CodeStyleVAPredictor:
    """
    使用 LLM 並以 instruct 風格 prompt 輸出 VA 數值。
    """

    def __init__(self, config: VAPredictorConfig) -> None:
        self.config = config

        adapter_path = Path(config.model_name_or_path)
        has_adapter = adapter_path.is_dir() and (adapter_path / "adapter_config.json").exists()
        self.adapter_path: Optional[str] = str(adapter_path) if has_adapter else None
        if has_adapter:
            peft_config = PeftConfig.from_pretrained(adapter_path)
            base_model_name = peft_config.base_model_name_or_path
            tokenizer_source = self.adapter_path
        else:
            peft_config = None
            base_model_name = config.model_name_or_path
            tokenizer_source = config.model_name_or_path

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, cache_dir=self.config.cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        if config.device:
            self.device = torch.device(config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_kwargs = {"trust_remote_code": True}
        if config.load_in_4bit or config.load_in_8bit:
            from transformers import BitsAndBytesConfig

            if config.load_in_4bit and config.load_in_8bit:
                raise ValueError("只能擇一使用 4bit 或 8bit 量化。")
            quant_kwargs = {"load_in_4bit": config.load_in_4bit, "load_in_8bit": config.load_in_8bit}
            if config.load_in_4bit:
                quant_kwargs.update(
                    {
                        "bnb_4bit_quant_type": "nf4",
                        "bnb_4bit_use_double_quant": True,
                        "bnb_4bit_compute_dtype": torch.bfloat16,
                    }
                )
            quant_config = BitsAndBytesConfig(**quant_kwargs)
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["device_map"] = "auto"
        else:
            if self.device.type == "cuda":
                model_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            **model_kwargs,
            cache_dir=self.config.cache_dir,
        )
        if not (config.load_in_4bit or config.load_in_8bit):
            self.model.to(self.device)

        if self.adapter_path:
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            if not (config.load_in_4bit or config.load_in_8bit):
                self.model.to(self.device)

        if config.load_in_4bit or config.load_in_8bit:
            self.model_device = next(self.model.parameters()).device
        else:
            self.model_device = self.device

        self.model.eval()

    _pattern_pair = re.compile(r"(-?\d+(?:\.\d+)?)\s*#\s*(-?\d+(?:\.\d+)?)")
    _pattern_number = re.compile(r"-?\d+(?:\.\d+)?")

    def parse_output(self, text: str) -> Optional[tuple[float, float]]:
        text = text.strip()
        match = self._pattern_pair.search(text)
        if match:
            try:
                valence = float(match.group(1))
                arousal = float(match.group(2))
                return clamp(valence), clamp(arousal)
            except ValueError:
                return None

        numbers = self._pattern_number.findall(text)
        if len(numbers) >= 2:
            try:
                valence = float(numbers[0])
                arousal = float(numbers[1])
                return clamp(valence), clamp(arousal)
            except ValueError:
                return None
        return None

    def predict(self, text: str, aspect: str, opinion: str) -> str:
        prompt = build_instruct_prompt(text, aspect, opinion)
        tokenized = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model_device) for k, v in tokenized.items()}
        max_tokens = max(1, min(self.config.max_new_tokens, 7))
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                # do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated[len(prompt) :]
        print("LLM 回應:", response)
        parsed = self.parse_output(response)
        print("LLM 解析結果:", parsed)
        if parsed is None:
            LOGGER.warning("LLM 未能解析輸出，改用 5.0#5.0。輸出原文: %s", response)
            return "5.0#5.0"
        valence, arousal = parsed
        return f"{valence:.1f}#{arousal:.1f}"


class DummyVAPredictor:
    """
    後備方案：若無法載入 LLM，提供中性分數。
    """

    def predict(self, text: str, aspect: str, opinion: str) -> str:
        return "5.0#5.0"
