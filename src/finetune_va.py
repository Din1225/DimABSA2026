"""
Fine-tune deepseek-7b-instruct-v1.5 LLM on VA regression with code-style prompts.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from trl import SFTConfig, SFTTrainer

from .data_utils import load_jsonl
from .va_prompts import build_instruct_prompt

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune LLM for VA regression.")
    parser.add_argument("--train-path", nargs="+", required=True, help="訓練 JSONL 檔案路徑，可以多個")
    parser.add_argument("--eval-path", nargs="+", help="驗證 JSONL 檔案路徑，可以多個")
    parser.add_argument("--output-dir", required=True, help="輸出資料夾")
    parser.add_argument(
        "--base-model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="預訓練模型名稱或本地路徑",
    )
    parser.add_argument("--cache-dir", help="HuggingFace 模型快取路徑")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--load-in-8bit", action="store_true", help="以 8bit 量化載入基礎模型")
    parser.add_argument("--no-4bit", action="store_true", help="停用預設的 4bit 量化")
    parser.add_argument("--bf16", action="store_true", help="啟用 bfloat16 訓練 (需 GPU 支援)")
    parser.add_argument("--fp16", action="store_true", help="啟用 fp16 訓練")
    return parser.parse_args()


def collect_examples(paths: Iterable[str]) -> List[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        for record in load_jsonl(path):
            text = record["Text"]
            for quad in record.get("Quadruplet", []):
                aspect = quad.get("Aspect", "").strip()
                opinion = quad.get("Opinion", "").strip()
                va = quad.get("VA", "").strip()
                if not aspect or not opinion or not va:
                    continue
                try:
                    valence_str, arousal_str = va.split("#")
                    valence = float(valence_str)
                    arousal = float(arousal_str)
                except ValueError:
                    LOGGER.warning("無法解析 VA: %s", va)
                    continue
                prompt = build_instruct_prompt(text, aspect, opinion)
                response = f"{valence:.1f}#{arousal:.1f}"
                rows.append({"text": f"{prompt}{response}"})
    return rows


def build_dataset(rows: List[dict[str, str]], seed: int) -> Dataset:
    if not rows:
        raise ValueError("提供的資料集中沒有有效的 quadruplet 樣本。")
    dataset = Dataset.from_list(rows)
    return dataset.shuffle(seed=seed)



def create_bnb_config(load_in_4bit: bool, load_in_8bit: bool) -> BitsAndBytesConfig | None:
    if load_in_4bit and load_in_8bit:
        raise ValueError("4bit 與 8bit 量化不能同時啟用。")
    if not load_in_4bit and not load_in_8bit:
        return None
    kwargs = {"load_in_4bit": load_in_4bit, "load_in_8bit": load_in_8bit}
    if load_in_4bit:
        kwargs.update(
            {
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_compute_dtype": torch.bfloat16,
            }
        )
    return BitsAndBytesConfig(**kwargs)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
    )


def main():
    args = parse_args()
    setup_logging()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = collect_examples(args.train_path)
    train_dataset = build_dataset(train_rows, args.seed)
    eval_dataset = None
    if args.eval_path:
        eval_rows = collect_examples(args.eval_path)
        if eval_rows:
            eval_dataset = build_dataset(eval_rows, args.seed)
        else:
            LOGGER.warning("驗證資料集中沒有有效樣本，將不執行評估。")

    
        
    



    load_in_4bit = not args.no_4bit
    load_in_8bit = args.load_in_8bit
    bnb_config = create_bnb_config(load_in_4bit, load_in_8bit)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    train_dataset = train_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
    )
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=["text"],
        )


    model_kwargs = {"trust_remote_code": True, "cache_dir": args.cache_dir}
    bf16_supported = torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    else:
        if torch.cuda.is_available():
            if args.bf16 and bf16_supported:
                model_kwargs["torch_dtype"] = torch.bfloat16
            elif args.fp16:
                model_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    elif torch.cuda.is_available():
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        remove_unused_columns=False,
        # dataset_text_field="text",
        packing=False,
        report_to=[],
        seed=args.seed,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        data_collator=data_collator,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    meta = {
        "base_model": args.base_model,
        "train_paths": args.train_path,
        "eval_paths": args.eval_path or [],
        "load_in_4bit": load_in_4bit,
        "load_in_8bit": load_in_8bit,
        "num_epochs": args.num_epochs,
        "max_length": args.max_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
    }
    with (output_dir / "va_finetune_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    LOGGER.info("LoRA 微調完成，權重儲存於 %s", output_dir)


if __name__ == "__main__":
    main()
