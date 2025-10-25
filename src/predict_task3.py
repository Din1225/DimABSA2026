"""
Subtask 3 推論腳本。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer

from .constants import INVALID_CATEGORY
from .data_utils import build_marked_text, load_jsonl, save_jsonl
from .inference_utils import assign_opinions_to_aspects, decode_bio
from .va_predictor import CodeStyleVAPredictor, DummyVAPredictor, VAPredictorConfig

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DimASQP prediction pipeline.")
    parser.add_argument("--model-root", required=True, help="訓練輸出根目錄")
    parser.add_argument("--input-path", required=True, help="輸入 JSONL 檔")
    parser.add_argument("--output-path", required=True, help="輸出 JSONL 檔")
    parser.add_argument("--device", default=None, help="推論裝置 (cpu / cuda)")
    parser.add_argument("--va-model-name", help="LLM 模型名稱或路徑")
    parser.add_argument("--va-max-new-tokens", type=int, default=8)
    parser.add_argument("--va-temperature", type=float, default=0.7)
    parser.add_argument("--va-top-p", type=float, default=0.9)
    parser.add_argument("--va-load-in-4bit", action="store_true", help="以 4bit 量化載入 LLM")
    parser.add_argument("--va-load-in-8bit", action="store_true", help="以 8bit 量化載入 LLM")
    parser.add_argument("--va-cache-dir", help="LLM cache 目錄")
    return parser.parse_args()


def get_id2label(config) -> dict[int, str]:
    raw = config.id2label
    if isinstance(raw, dict):
        return {int(k): v for k, v in raw.items()}
    return {idx: label for idx, label in enumerate(raw)}


def build_predictor(args: argparse.Namespace):
    if args.va_model_name:
        config = VAPredictorConfig(
            model_name_or_path=args.va_model_name,
            device=args.device,
            max_new_tokens=args.va_max_new_tokens,
            temperature=args.va_temperature,
            top_p=args.va_top_p,
            load_in_4bit=args.va_load_in_4bit,
            load_in_8bit=args.va_load_in_8bit,
            cache_dir=args.va_cache_dir,
        )
        LOGGER.info("使用 LLM 進行 VA 回歸：%s", args.va_model_name)
        return CodeStyleVAPredictor(config)
    LOGGER.warning("未指定 LLM，使用 DummyVAPredictor (固定 5.0#5.0)")
    return DummyVAPredictor()


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device):
    return {k: v.to(device) for k, v in batch.items()}


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_root = Path(args.model_root)
    meta_path = model_root / "metadata.json"
    relation_dir = model_root / "relation_classifier"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        aspect_dir = Path(meta["aspect_dir"])
        opinion_dir = Path(meta["opinion_dir"])
        category_dir = Path(meta.get("category_dir", model_root / "category_classifier"))
        relation_dir = Path(meta.get("relation_dir", relation_dir))
    else:
        aspect_dir = model_root / "aspect_extractor"
        opinion_dir = model_root / "opinion_extractor"
        category_dir = model_root / "category_classifier"
        relation_dir = model_root / "relation_classifier"

    LOGGER.info("載入 Aspect 抽取模型：%s", aspect_dir)
    aspect_model = AutoModelForTokenClassification.from_pretrained(aspect_dir).to(device)
    aspect_model.eval()
    aspect_tokenizer = AutoTokenizer.from_pretrained(aspect_dir)
    aspect_id2label = get_id2label(aspect_model.config)

    LOGGER.info("載入 Opinion 抽取模型：%s", opinion_dir)
    opinion_model = AutoModelForTokenClassification.from_pretrained(opinion_dir).to(device)
    opinion_model.eval()
    opinion_tokenizer = AutoTokenizer.from_pretrained(opinion_dir)
    opinion_id2label = get_id2label(opinion_model.config)

    relation_model = None
    relation_tokenizer = None
    relation_id2label: dict[int, str] | None = None
    if relation_dir.exists():
        LOGGER.info("載入 Relation 分類模型：%s", relation_dir)
        relation_model = AutoModelForSequenceClassification.from_pretrained(relation_dir).to(device)
        relation_model.eval()
        relation_tokenizer = AutoTokenizer.from_pretrained(relation_dir)
        relation_id2label = get_id2label(relation_model.config)
    else:
        LOGGER.warning("找不到 Relation 分類模型，將使用既有 Category pipeline：%s", relation_dir)

    category_model = None
    category_tokenizer = None
    category_id2label: dict[int, str] | None = None
    if category_dir.exists():
        LOGGER.info("載入 Category 分類模型：%s", category_dir)
        category_model = AutoModelForSequenceClassification.from_pretrained(category_dir).to(device)
        category_model.eval()
        category_tokenizer = AutoTokenizer.from_pretrained(category_dir)
        category_id2label = get_id2label(category_model.config)
    else:
        LOGGER.warning("找不到 Category 分類模型，fallback 可能失效：%s", category_dir)

    va_predictor = build_predictor(args)

    inputs = load_jsonl(args.input_path)
    outputs = []
    cache = {}
    for sample in inputs:
        text = sample["Text"]
        # Aspect
        aspect_encoded = aspect_tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            return_tensors="pt",
        )
        aspect_offsets = aspect_encoded.pop("offset_mapping")[0].tolist()
        aspect_inputs = move_to_device(aspect_encoded, device)
        with torch.no_grad():
            aspect_logits = aspect_model(**aspect_inputs).logits
        aspect_pred = aspect_logits.argmax(dim=-1)[0].tolist()
        aspects = decode_bio(aspect_pred, aspect_offsets, aspect_id2label, "B-ASPECT", "I-ASPECT", text)

        # Opinion
        opinion_encoded = opinion_tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            return_tensors="pt",
        )
        opinion_offsets = opinion_encoded.pop("offset_mapping")[0].tolist()
        opinion_inputs = move_to_device(opinion_encoded, device)
        with torch.no_grad():
            opinion_logits = opinion_model(**opinion_inputs).logits
        opinion_pred = opinion_logits.argmax(dim=-1)[0].tolist()
        opinions = decode_bio(opinion_pred, opinion_offsets, opinion_id2label, "B-OPINION", "I-OPINION", text)

        if opinions and not aspects:
            aspects = [{"text": text, "start": 0, "end": len(text)}]
        if aspects and not opinions:
            opinions = [{"text": text, "start": 0, "end": len(text)}]

        def run_category_fallback() -> list[tuple[dict[str, int | str], dict[str, int | str], str]]:
            if category_model is None or category_tokenizer is None or category_id2label is None:
                return []
            aspect_categories: dict[tuple[str, int, int], str] = {}
            for asp in aspects:
                key = (asp["text"], int(asp["start"]), int(asp["end"]))
                encoded = category_tokenizer(
                    asp["text"],
                    text,
                    truncation=True,
                    return_tensors="pt",
                )
                encoded = move_to_device(encoded, device)
                with torch.no_grad():
                    logits = category_model(**encoded).logits
                label_id = int(logits.argmax(dim=-1).item())
                category = category_id2label[label_id]
                aspect_categories[key] = category
            pairs = []
            for asp, opn in assign_opinions_to_aspects(aspects, opinions):
                key = (asp["text"], int(asp["start"]), int(asp["end"]))
                pairs.append((asp, opn, aspect_categories.get(key, "LAPTOP#GENERAL")))
            return pairs

        candidate_pairs: list[tuple[dict[str, int | str], dict[str, int | str], str]] = []
        if relation_model is not None and relation_tokenizer is not None and relation_id2label is not None:
            sep = relation_tokenizer.sep_token or "[SEP]"
            seen_keys: set[tuple[int, int, int, int, str]] = set()
            for asp in aspects:
                for opn in opinions:
                    marked = build_marked_text(text, asp, opn)
                    pair_repr = f"{asp['text']} {sep} {opn['text']}".strip()
                    encoded = relation_tokenizer(
                        marked,
                        pair_repr,
                        truncation=True,
                        return_tensors="pt",
                    )
                    encoded = move_to_device(encoded, device)
                    with torch.no_grad():
                        logits = relation_model(**encoded).logits
                    label_id = int(logits.argmax(dim=-1).item())
                    label = relation_id2label[label_id]
                    key = (
                        int(asp["start"]),
                        int(asp["end"]),
                        int(opn["start"]),
                        int(opn["end"]),
                        label,
                    )
                    if label == INVALID_CATEGORY:
                        continue
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    candidate_pairs.append((asp, opn, label))
            if not candidate_pairs:
                LOGGER.debug("Relation 模型未產生有效配對，啟用 fallback 類別分類")
                candidate_pairs = run_category_fallback()
        else:
            candidate_pairs = run_category_fallback()

        quadruplets = []
        for asp, opn, category in candidate_pairs:
            cache_key = (text, asp["text"], opn["text"])
            if cache_key not in cache:
                cache[cache_key] = va_predictor.predict(text, asp["text"], opn["text"])
            va_string = cache[cache_key]
            quadruplets.append(
                {
                    "Aspect": asp["text"],
                    "Category": category,
                    "Opinion": opn["text"],
                    "VA": va_string,
                }
            )
        outputs.append({"ID": sample["ID"], "Quadruplet": quadruplets})

    save_jsonl(args.output_path, outputs)
    LOGGER.info("推論完成，結果寫入 %s", args.output_path)


if __name__ == "__main__":
    main()
