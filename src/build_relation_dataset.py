"""以論文方法建立 Aspect-Opinion relation 訓練資料（含負樣本）。"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterable, List

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .constants import ASPECT_TAGS, OPINION_TAGS, INVALID_CATEGORY
from .data_utils import (
    collect_aspect_spans,
    collect_opinion_spans,
    load_jsonl,
    save_jsonl,
)
from .inference_utils import decode_bio

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build relation dataset with negative pairs via k-fold CV.")
    parser.add_argument("--input-path", required=True, help="原始 JSONL 訓練資料")
    parser.add_argument("--output-path", required=True, help="輸出配對資料 JSONL 路徑")
    parser.add_argument("--model-name", default="bert-base-chinese", help="預訓練權重名稱或路徑")
    parser.add_argument("--num-folds", type=int, default=5, help="K-fold 折數")
    parser.add_argument("--num-epochs", type=int, default=3, help="折內訓練 epoch 數")
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-neg-per-sample", type=int, default=10, help="每句最多加入多少負樣本")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def split_folds(samples: List[dict[str, Any]], num_folds: int, seed: int) -> List[List[int]]:
    indices = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    folds: List[List[int]] = [[] for _ in range(num_folds)]
    for idx, original in enumerate(indices):
        folds[idx % num_folds].append(original)
    return folds


def make_training_args(tmp_dir: Path, args: argparse.Namespace) -> TrainingArguments:
    no_cuda = args.device != "cuda"
    return TrainingArguments(
        output_dir=str(tmp_dir),
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.train_batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        gradient_accumulation_steps=1,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        optim="adamw_torch",
        no_cuda=no_cuda,
    )


def train_token_classifier(
    model_name: str,
    tokenizer,
    labels: List[str],
    train_samples: List[dict[str, Any]],
    task: str,
    args: argparse.Namespace,
) -> tuple[AutoModelForTokenClassification, dict[int, str]]:
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    dataset = TokenTaggingDataset(train_samples, tokenizer, label2id, task=task, max_length=args.max_length)
    if len(dataset) == 0:
        raise ValueError(f"{task} 資料為空，無法訓練")
    with TemporaryDirectory() as tmp_dir:
        training_args = make_training_args(Path(tmp_dir), args)
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )
        collator = DataCollatorForTokenClassification(tokenizer)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collator,
            tokenizer=tokenizer,
        )
        trainer.train()
        model = trainer.model
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, id2label


def predict_spans(
    model: AutoModelForTokenClassification,
    tokenizer,
    samples: Iterable[dict[str, Any]],
    id2label: dict[int, str],
    task: str,
    device: torch.device,
    max_length: int,
) -> list[list[dict[str, Any]]]:
    begin_tag = "B-ASPECT" if task == "aspect" else "B-OPINION"
    inside_tag = "I-ASPECT" if task == "aspect" else "I-OPINION"
    preds: list[list[dict[str, Any]]] = []
    for sample in samples:
        text = sample["Text"]
        encoded = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
        label_ids = logits.argmax(dim=-1)[0].tolist()
        spans = decode_bio(label_ids, offsets, id2label, begin_tag, inside_tag, text)
        preds.append(spans)
    return preds


class TokenTaggingDataset(torch.utils.data.Dataset):
    """重新引用現有 TokenTaggingDataset，避免循環匯入。"""

    def __init__(
        self,
        samples: List[dict[str, Any]],
        tokenizer,
        label2id: dict[str, int],
        task: str,
        max_length: int,
    ) -> None:
        from .datasets import TokenTaggingDataset as _TokenTaggingDataset

        self._inner = _TokenTaggingDataset(samples, tokenizer, label2id, task=task, max_length=max_length)

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int):
        return self._inner[idx]


def convert_span(span: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": span.get("text", ""),
        "start": int(span.get("start", -1)),
        "end": int(span.get("end", -1)),
    }


def make_pair_entry(
    sample_id: str,
    text: str,
    aspect: dict[str, Any],
    opinion: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    return {
        "ID": sample_id,
        "Text": text,
        "Aspect": convert_span(aspect),
        "Opinion": convert_span(opinion),
        "Label": label,
    }


def build_pairs(
    samples: List[dict[str, Any]],
    predictions: dict[str, dict[str, list[dict[str, Any]]]],
    max_neg_per_sample: int,
    seed: int,
) -> List[dict[str, Any]]:
    rng = random.Random(seed)
    all_pairs: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = sample["ID"]
        text = sample["Text"]
        quads = sample.get("Quadruplet", []) or []

        gold_aspects = collect_aspect_spans(text, quads)
        gold_opinions = collect_opinion_spans(text, quads)

        gold_pairs = set()
        for idx, quad in enumerate(quads):
            if idx >= len(gold_aspects) or idx >= len(gold_opinions):
                continue
            label = quad.get("Category", "").strip().upper()
            aspect_span = gold_aspects[idx]
            opinion_span = gold_opinions[idx]
            entry = make_pair_entry(sample_id, text, aspect_span, opinion_span, label)
            all_pairs.append(entry)
            key = (
                int(aspect_span.get("start", -1)),
                int(aspect_span.get("end", -1)),
                int(opinion_span.get("start", -1)),
                int(opinion_span.get("end", -1)),
            )
            gold_pairs.add(key)

        pred_info = predictions.get(sample_id, {"aspects": [], "opinions": []})
        pred_aspects = [convert_span(span) for span in pred_info.get("aspects", [])]
        pred_opinions = [convert_span(span) for span in pred_info.get("opinions", [])]

        candidate_negatives: list[dict[str, Any]] = []

        def enqueue_negative(asp: dict[str, Any], opn: dict[str, Any]):
            if not asp or not opn:
                return
            key = (
                int(asp.get("start", -1)),
                int(asp.get("end", -1)),
                int(opn.get("start", -1)),
                int(opn.get("end", -1)),
            )
            if key in gold_pairs:
                return
            entry = make_pair_entry(sample_id, text, asp, opn, INVALID_CATEGORY)
            candidate_negatives.append(entry)

        # 預測-預測
        for asp in pred_aspects:
            for opn in pred_opinions:
                enqueue_negative(asp, opn)

        # 預測 aspect + 正確 opinion
        for asp in pred_aspects:
            for opn in gold_opinions:
                enqueue_negative(asp, convert_span(opn))

        # 正確 aspect + 預測 opinion
        for asp in gold_aspects:
            for opn in pred_opinions:
                enqueue_negative(convert_span(asp), opn)

        if candidate_negatives:
            rng.shuffle(candidate_negatives)
            all_pairs.extend(candidate_negatives[:max_neg_per_sample])

    return all_pairs


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    set_seed(args.seed)

    samples = load_jsonl(args.input_path)
    if args.num_folds < 2:
        raise ValueError("num_folds 至少為 2")
    folds = split_folds(samples, args.num_folds, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    aspect_predictions: dict[str, list[dict[str, Any]]] = {}
    opinion_predictions: dict[str, list[dict[str, Any]]] = {}

    for fold_id, val_indices in enumerate(folds):
        LOGGER.info("Fold %d/%d - 驗證樣本數: %d", fold_id + 1, args.num_folds, len(val_indices))
        train_indices = [idx for idx in range(len(samples)) if idx not in val_indices]
        train_split = [samples[idx] for idx in train_indices]
        val_split = [samples[idx] for idx in val_indices]

        aspect_model, aspect_id2label = train_token_classifier(
            args.model_name,
            tokenizer,
            ASPECT_TAGS,
            train_split,
            task="aspect",
            args=args,
        )
        opinion_model, opinion_id2label = train_token_classifier(
            args.model_name,
            tokenizer,
            OPINION_TAGS,
            train_split,
            task="opinion",
            args=args,
        )

        aspect_spans = predict_spans(aspect_model, tokenizer, val_split, aspect_id2label, "aspect", device, args.max_length)
        opinion_spans = predict_spans(opinion_model, tokenizer, val_split, opinion_id2label, "opinion", device, args.max_length)

        for idx, sample_idx in enumerate(val_indices):
            sample_id = samples[sample_idx]["ID"]
            aspect_predictions[sample_id] = aspect_spans[idx]
            opinion_predictions[sample_id] = opinion_spans[idx]

        # 釋放資源
        del aspect_model
        del opinion_model
        torch.cuda.empty_cache()

    merged_predictions: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for sample in samples:
        sample_id = sample["ID"]
        merged_predictions[sample_id] = {
            "aspects": aspect_predictions.get(sample_id, []),
            "opinions": opinion_predictions.get(sample_id, []),
        }

    pairs = build_pairs(samples, merged_predictions, args.max_neg_per_sample, args.seed)
    LOGGER.info("最終 pair 數量：%d", len(pairs))
    save_jsonl(args.output_path, pairs)
    LOGGER.info("資料已寫入 %s", args.output_path)


if __name__ == "__main__":
    main()
