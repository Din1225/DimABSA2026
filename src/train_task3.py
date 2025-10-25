"""
Subtask 3 訓練腳本：BERT pipeline
- Aspect 抽取 (token classification)
- Opinion 抽取 (token classification)
- Aspect Category 分類
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .constants import (
    ASPECT_TAGS,
    OPINION_TAGS,
    CATEGORY2ID,
    ID2CATEGORY,
    RELATION2ID,
    ID2RELATION,
)
from .data_utils import load_jsonl
from .datasets import AspectCategoryDataset, AspectOpinionPairDataset, TokenTaggingDataset

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DimASQP pipeline models.")
    parser.add_argument("--train-path", required=True, help="訓練資料 JSONL 路徑")
    parser.add_argument("--dev-path", help="驗證資料 JSONL 路徑")
    parser.add_argument("--output-dir", required=True, help="輸出根目錄")
    parser.add_argument("--model-name", default="bert-base-chinese", help="預訓練權重名稱或路徑")
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--train-pair-path", required=True, help="Aspect-Opinion relation 訓練資料 JSONL")
    parser.add_argument("--dev-pair-path", help="Aspect-Opinion relation 驗證資料 JSONL")
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
    )


def train_token_classifier(
    model_name: str,
    task_name: str,
    tokenizer,
    labels: list[str],
    train_data,
    dev_data,
    training_args: TrainingArguments,
    max_length: int,
) -> None:
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    train_dataset = TokenTaggingDataset(train_data, tokenizer, label2id, task=task_name, max_length=max_length)
    if len(train_dataset) == 0:
        LOGGER.warning("%s 訓練資料為空，跳過", task_name)
        return
    eval_dataset = (
        TokenTaggingDataset(dev_data, tokenizer, label2id, task=task_name, max_length=max_length)
        if dev_data is not None
        else None
    )
    collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    with Path(training_args.output_dir, "label2id.json").open("w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)


def train_category_classifier(
    model_name: str,
    tokenizer,
    train_data,
    dev_data,
    training_args: TrainingArguments,
    max_length: int,
) -> None:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(CATEGORY2ID),
        id2label=ID2CATEGORY,
        label2id=CATEGORY2ID,
    )
    train_dataset = AspectCategoryDataset(train_data, tokenizer, CATEGORY2ID, max_length=max_length)
    if len(train_dataset) == 0:
        LOGGER.warning("Category 訓練資料為空，跳過")
        return
    eval_dataset = (
        AspectCategoryDataset(dev_data, tokenizer, CATEGORY2ID, max_length=max_length)
        if dev_data is not None
        else None
    )
    collator = DataCollatorWithPadding(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


def train_relation_classifier(
    model_name: str,
    tokenizer,
    train_data,
    dev_data,
    training_args: TrainingArguments,
    max_length: int,
) -> None:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(RELATION2ID),
        id2label=ID2RELATION,
        label2id=RELATION2ID,
    )
    train_dataset = AspectOpinionPairDataset(train_data, tokenizer, RELATION2ID, max_length=max_length)
    if len(train_dataset) == 0:
        LOGGER.warning("Relation 訓練資料為空，跳過")
        return
    eval_dataset = (
        AspectOpinionPairDataset(dev_data, tokenizer, RELATION2ID, max_length=max_length)
        if dev_data is not None
        else None
    )
    collator = DataCollatorWithPadding(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    label_path = Path(training_args.output_dir, "relation_label2id.json")
    with label_path.open("w", encoding="utf-8") as f:
        json.dump(RELATION2ID, f, ensure_ascii=False, indent=2)


def load_splits(train_path: str, dev_path: Optional[str]):
    train_data = load_jsonl(train_path)
    dev_data = load_jsonl(dev_path) if dev_path else None
    LOGGER.info("Train size: %d", len(train_data))
    if dev_data is not None:
        LOGGER.info("Dev size: %d", len(dev_data))
    return train_data, dev_data


def make_training_args(
    output_dir: Path,
    num_train_epochs: int,
    train_batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    do_eval: bool,
):
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=1,
        report_to=[],
        gradient_accumulation_steps=1,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        max_grad_norm=1,
        load_best_model_at_end=False,
        metric_for_best_model=None,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        label_smoothing_factor=0.0,
        optim="adamw_torch",
    )


def main():
    args = parse_args()
    setup_logging()
    set_seed(args.seed)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    aspect_dir = output_root / "aspect_extractor"
    opinion_dir = output_root / "opinion_extractor"
    category_dir = output_root / "category_classifier"
    relation_dir = output_root / "relation_classifier"
    aspect_dir.mkdir(exist_ok=True)
    opinion_dir.mkdir(exist_ok=True)
    category_dir.mkdir(exist_ok=True)
    relation_dir.mkdir(exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_data, dev_data = load_splits(args.train_path, args.dev_path)
    train_pair_data = load_jsonl(args.train_pair_path)
    dev_pair_data = load_jsonl(args.dev_pair_path) if args.dev_pair_path else None

    LOGGER.info("開始訓練 Aspect 抽取模型")
    has_dev = dev_data is not None

    aspect_args = make_training_args(
        aspect_dir,
        args.num_epochs,
        args.train_batch_size,
        args.eval_batch_size,
        args.learning_rate,
        args.weight_decay,
        has_dev,
    )
    train_token_classifier(
        args.model_name,
        "aspect",
        tokenizer,
        ASPECT_TAGS,
        train_data,
        dev_data,
        aspect_args,
        args.max_length,
    )

    LOGGER.info("開始訓練 Opinion 抽取模型")
    opinion_args = make_training_args(
        opinion_dir,
        args.num_epochs,
        args.train_batch_size,
        args.eval_batch_size,
        args.learning_rate,
        args.weight_decay,
        has_dev,
    )
    train_token_classifier(
        args.model_name,
        "opinion",
        tokenizer,
        OPINION_TAGS,
        train_data,
        dev_data,
        opinion_args,
        args.max_length,
    )

    LOGGER.info("開始訓練 Aspect Category 分類模型")
    category_args = make_training_args(
        category_dir,
        args.num_epochs,
        args.train_batch_size,
        args.eval_batch_size,
        args.learning_rate,
        args.weight_decay,
        has_dev,
    )
    train_category_classifier(args.model_name, tokenizer, train_data, dev_data, category_args, args.max_length)

    LOGGER.info("開始訓練 Relation 分類模型")
    relation_args = make_training_args(
        relation_dir,
        args.num_epochs,
        args.train_batch_size,
        args.eval_batch_size,
        args.learning_rate,
        args.weight_decay,
        dev_pair_data is not None,
    )
    train_relation_classifier(
        args.model_name,
        tokenizer,
        train_pair_data,
        dev_pair_data,
        relation_args,
        args.max_length,
    )

    meta = {
        "model_name": args.model_name,
        "aspect_dir": str(aspect_dir),
        "opinion_dir": str(opinion_dir),
        "category_dir": str(category_dir),
        "relation_dir": str(relation_dir),
    }
    with (output_root / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    LOGGER.info("訓練完成，模型已儲存至 %s", output_root)


if __name__ == "__main__":
    main()
