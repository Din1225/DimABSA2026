"""
資料處理工具：讀寫 JSONL 與標註 span。
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Dict, Any

LOGGER = logging.getLogger(__name__)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """讀取 JSON Lines 檔案。"""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """輸出 JSON Lines 檔案。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _find_all_occurrences(text: str, target: str) -> list[tuple[int, int]]:
    """回傳 target 在 text 中所有出現位置的 (start, end) 座標。"""
    if not target:
        return []
    matches = [m.span() for m in re.finditer(re.escape(target), text)]
    if matches:
        return matches
    # 若直接比對失敗，試圖去除空白重新比對
    compact_text = text.replace(" ", "")
    compact_target = target.replace(" ", "")
    if compact_text == text or compact_target == target:
        return []
    matches = [m.span() for m in re.finditer(re.escape(compact_target), compact_text)]
    if not matches:
        return []
    # 將去空白後的位置對映回原字串（保守策略：回傳最靠近的原始索引）
    reconstructed: list[tuple[int, int]] = []
    pointer = 0
    for start_c, end_c in matches:
        # 尋找對應回原產生的索引
        start = _map_compact_index(text, start_c)
        end = _map_compact_index(text, end_c - 1) + 1
        reconstructed.append((start, end))
        pointer = end
    return reconstructed


def _map_compact_index(text: str, compact_idx: int) -> int:
    """將去除空白後的索引位置映射回原始字串索引。"""
    count = -1
    for idx, ch in enumerate(text):
        if ch != " ":
            count += 1
        if count == compact_idx:
            return idx
    return len(text) - 1


def extract_spans_from_quads(
    text: str, quads: Iterable[dict[str, Any]], key: str
) -> list[dict[str, Any]]:
    """
    從 quadruplet 陣列取得特定欄位 (Aspect 或 Opinion) 的 span。
    回傳格式：[{ "text": str, "start": int, "end": int }]
    """
    assert key in {"Aspect", "Opinion"}
    seen_counter: dict[str, int] = defaultdict(int)
    spans: list[dict[str, Any]] = []
    for quad in quads:
        mention = quad.get(key, "")
        mention = mention.strip()
        if not mention:
            continue
        occurrences = _find_all_occurrences(text, mention)
        if not occurrences:
            LOGGER.warning("找不到 %s: %s in text: %s", key, mention, text)
            continue
        use_idx = seen_counter[(mention, key)]
        if use_idx >= len(occurrences):
            use_idx = len(occurrences) - 1
        start, end = occurrences[use_idx]
        spans.append({"text": mention, "start": start, "end": end})
        seen_counter[(mention, key)] += 1
    return spans


def collect_aspect_spans(text: str, quads: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return extract_spans_from_quads(text, quads, "Aspect")


def collect_opinion_spans(text: str, quads: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return extract_spans_from_quads(text, quads, "Opinion")

