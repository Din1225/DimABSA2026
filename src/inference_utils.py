"""
推論輔助：BIO 標籤解碼、距離配對等。
"""

from __future__ import annotations

from typing import Iterable, List, Tuple


def decode_bio(
    label_ids: Iterable[int],
    offsets: Iterable[Tuple[int, int]],
    id2label: dict[int, str],
    begin_tag: str,
    inside_tag: str,
    text: str,
) -> list[dict[str, int | str]]:
    """將 BIO 預測轉回 span。"""
    spans: list[dict[str, int | str]] = []
    active_start: int | None = None
    active_end: int | None = None
    for idx, (label_id, (start, end)) in enumerate(zip(label_ids, offsets)):
        if start == end == 0:
            continue
        tag = id2label[label_id]
        if tag == begin_tag:
            if active_start is not None:
                spans.append(
                    {"start": active_start, "end": active_end, "text": text[active_start:active_end]}
                )
            active_start = start
            active_end = end
        elif tag == inside_tag and active_start is not None:
            active_end = end
        else:
            if active_start is not None:
                spans.append(
                    {"start": active_start, "end": active_end, "text": text[active_start:active_end]}
                )
                active_start = None
                active_end = None
    if active_start is not None:
        spans.append({"start": active_start, "end": active_end, "text": text[active_start:active_end]})
    return spans


def assign_opinions_to_aspects(
    aspects: list[dict[str, int | str]], opinions: list[dict[str, int | str]]
) -> list[tuple[dict[str, int | str], dict[str, int | str]]]:
    """
    將每個意見對應到最近的 aspect。
    若無 aspect，回傳空陣列。
    """
    if not aspects or not opinions:
        return []
    pairs = []
    for opinion in opinions:
        opinion_center = (int(opinion["start"]) + int(opinion["end"])) / 2.0
        best_aspect = min(
            aspects,
            key=lambda asp: abs(((int(asp["start"]) + int(asp["end"])) / 2.0) - opinion_center),
        )
        pairs.append((best_aspect, opinion))
    return pairs

