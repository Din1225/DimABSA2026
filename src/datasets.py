"""
Dataset 物件與前處理函式。
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from .data_utils import build_marked_text, collect_aspect_spans, collect_opinion_spans


def _encode_token_labels(
    encodings: Dict[str, Any],
    spans: List[dict[str, Any]],
    label2id: dict[str, int],
    begin_tag: str,
    inside_tag: str,
) -> List[int]:
    labels = [label2id["O"]] * len(encodings["offset_mapping"])
    for span in spans:
        span_start = span["start"]
        span_end = span["end"]
        has_started = False
        for idx, (start, end) in enumerate(encodings["offset_mapping"]):
            if start == end == 0:
                continue  # special tokens
            if end <= span_start or start >= span_end:
                continue
            if not has_started:
                labels[idx] = label2id[begin_tag]
                has_started = True
            else:
                labels[idx] = label2id[inside_tag]
    return labels


class TokenTaggingDataset(Dataset):
    """
    子任務：Aspect 或 Opinion 抽取。輸出 BIO 標籤。
    """

    def __init__(
        self,
        samples: List[dict[str, Any]],
        tokenizer,
        label2id: dict[str, int],
        task: str,
        max_length: int = 256,
    ) -> None:
        assert task in {"aspect", "opinion"}
        self.encodings: List[Dict[str, Any]] = []
        self.labels: List[List[int]] = []
        begin_tag = "B-ASPECT" if task == "aspect" else "B-OPINION"
        inside_tag = "I-ASPECT" if task == "aspect" else "I-OPINION"
        span_func = collect_aspect_spans if task == "aspect" else collect_opinion_spans
        for sample in samples:
            text = sample["Text"]
            quads = sample.get("Quadruplet", [])
            spans = span_func(text, quads)
            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=max_length,
            )
            labels = _encode_token_labels(enc, spans, label2id, begin_tag, inside_tag)
            enc.pop("offset_mapping")
            self.encodings.append({k: torch.tensor(v) for k, v in enc.items()})
            self.labels.append(torch.tensor(labels, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {k: v.clone().detach() for k, v in self.encodings[idx].items()}
        item["labels"] = self.labels[idx].clone().detach()
        return item


class AspectCategoryDataset(Dataset):
    """
    子任務：Aspect Category 分類。輸出類別索引。
    """

    def __init__(
        self,
        samples: List[dict[str, Any]],
        tokenizer,
        label2id: dict[str, int],
        max_length: int = 256,
    ) -> None:
        self.encodings: List[Dict[str, Any]] = []
        self.labels: List[int] = []
        for sample in samples:
            text = sample["Text"]
            quads = sample.get("Quadruplet", [])
            for quad in quads:
                aspect = quad["Aspect"].strip()
                category = quad["Category"].strip().upper()
                if category not in label2id:
                    continue
                enc = tokenizer(
                    aspect,
                    text,
                    truncation=True,
                    max_length=max_length,
                )
                self.encodings.append({k: torch.tensor(v) for k, v in enc.items()})
                self.labels.append(label2id[category])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {k: v.clone().detach() for k, v in self.encodings[idx].items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class AspectOpinionPairDataset(Dataset):
    """Aspect-Opinion relation 資料集（含 Invalid 負樣本）。"""

    def __init__(
        self,
        samples: List[dict[str, Any]],
        tokenizer,
        label2id: dict[str, int],
        max_length: int = 256,
    ) -> None:
        self.encodings: List[Dict[str, Any]] = []
        self.labels: List[int] = []
        sep = tokenizer.sep_token or "[SEP]"
        for sample in samples:
            text = sample["Text"]
            aspect_span = sample.get("Aspect") or {}
            opinion_span = sample.get("Opinion") or {}
            label = (sample.get("Label") or "").strip().upper()
            if label not in label2id:
                continue
            aspect_text = aspect_span.get("text", "").strip()
            opinion_text = opinion_span.get("text", "").strip()
            marked = build_marked_text(text, aspect_span, opinion_span)
            pair_repr = f"{aspect_text} {sep} {opinion_text}".strip()
            enc = tokenizer(
                marked,
                pair_repr,
                truncation=True,
                max_length=max_length,
            )
            self.encodings.append({k: torch.tensor(v) for k, v in enc.items()})
            self.labels.append(label2id[label])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {k: v.clone().detach() for k, v in self.encodings[idx].items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
