import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

NULL_TOKEN = "[NULL]"
NULL_TEXT = "NULL"


@dataclass
class Quadruplet:
    aspect: str
    category: str
    opinion: str
    valence: float
    arousal: float


@dataclass
class SentenceInstance:
    idx: str
    text: str
    quadruplets: List[Quadruplet]


def _parse_va(va_str: str) -> Tuple[float, float]:
    valence_str, arousal_str = va_str.split("#")
    return float(valence_str), float(arousal_str)


def load_jsonl(path: Path) -> List[SentenceInstance]:
    instances: List[SentenceInstance] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            payload = json.loads(line)
            quadruplets: List[Quadruplet] = []
            for quad in payload.get("Quadruplet", []):
                valence, arousal = _parse_va(quad["VA"])
                quadruplets.append(
                    Quadruplet(
                        aspect=quad["Aspect"],
                        category=quad["Category"],
                        opinion=quad["Opinion"],
                        valence=valence,
                        arousal=arousal,
                    )
                )
            instances.append(
                SentenceInstance(
                    idx=payload["ID"],
                    text=payload["Text"],
                    quadruplets=quadruplets,
                )
            )
    return instances


def train_valid_split(
    instances: Sequence[SentenceInstance],
    valid_ratio: float,
    seed: int,
) -> Tuple[List[SentenceInstance], List[SentenceInstance]]:
    n_total = len(instances)
    n_valid = max(1, int(n_total * valid_ratio))
    rng = random.Random(seed)
    indices = list(range(n_total))
    rng.shuffle(indices)
    valid_indices = set(indices[:n_valid])
    train_set: List[SentenceInstance] = []
    valid_set: List[SentenceInstance] = []
    for i, inst in enumerate(instances):
        if i in valid_indices:
            valid_set.append(inst)
        else:
            train_set.append(inst)
    return train_set, valid_set


def collect_categories(instances: Iterable[SentenceInstance]) -> List[str]:
    categories = set()
    for inst in instances:
        for quad in inst.quadruplets:
            categories.add(quad.category)
    return sorted(categories)


def find_all_occurrences(text: str, target: str) -> List[Tuple[int, int]]:
    if not target:
        return []
    matches: List[Tuple[int, int]] = []
    start = text.find(target, 0)
    while start != -1:
        matches.append((start, start + len(target)))
        start = text.find(target, start + 1)
    return matches


def assign_spans(text: str, items: Sequence[str]) -> Dict[str, List[Tuple[int, int]]]:
    span_usage: Dict[Tuple[int, int], bool] = {}
    result: Dict[str, List[Tuple[int, int]]] = {}
    for item in items:
        if item == NULL_TEXT:
            result.setdefault(item, []).append((-1, -1))
            continue
        matches = find_all_occurrences(text, item)
        chosen: Optional[Tuple[int, int]] = None
        for span in matches:
            if not span_usage.get(span, False):
                span_usage[span] = True
                chosen = span
                break
        if chosen is None and matches:
            chosen = matches[0]
        if chosen is None:
            continue
        result.setdefault(item, []).append(chosen)
    return result


def sentence_to_char_labels(
    text: str,
    quadruplets: Sequence[Quadruplet],
    prefix: str,
) -> Tuple[List[int], List[int]]:
    total_len = len(prefix) + len(text)
    aspect_labels = [0] * total_len
    opinion_labels = [0] * total_len
    aspects = [quad.aspect for quad in quadruplets if quad.aspect != NULL_TEXT]
    opinions = [quad.opinion for quad in quadruplets if quad.opinion != NULL_TEXT]
    aspect_spans = assign_spans(text, aspects)
    opinion_spans = assign_spans(text, opinions)

    def mark(spans: Dict[str, List[Tuple[int, int]]], labels: List[int]) -> None:
        for entries in spans.values():
            for start, end in entries:
                start += len(prefix)
                end += len(prefix)
                if start < 0 or end <= start:
                    continue
                labels[start] = 1
                for pos in range(start + 1, min(end, len(labels))):
                    labels[pos] = 2

    mark(aspect_spans, aspect_labels)
    mark(opinion_spans, opinion_labels)

    if prefix:
        if any(quad.aspect == NULL_TEXT for quad in quadruplets):
            aspect_labels[0] = 1
            for pos in range(1, min(len(prefix), len(aspect_labels))):
                aspect_labels[pos] = 2
        if any(quad.opinion == NULL_TEXT for quad in quadruplets):
            opinion_labels[0] = 1
            for pos in range(1, min(len(prefix), len(opinion_labels))):
                opinion_labels[pos] = 2
    return aspect_labels, opinion_labels


def get_valence_arousal_bins(value: float, step: float = 0.25) -> int:
    value = min(9.0, max(1.0, value))
    return int(round((value - 1.0) / step))


def get_bin_center(bin_id: int, step: float = 0.25) -> float:
    return round(1.0 + bin_id * step, 2)
