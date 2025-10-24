"""
VA 後處理：將 LLM 輸出的 V#A 字串統一補齊到小數點兩位。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data_utils import load_jsonl, save_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process VA predictions to two decimals.")
    parser.add_argument("--input-path", required=True, help="原始預測 JSONL 路徑")
    parser.add_argument("--output-path", required=True, help="修正後輸出的 JSONL 路徑")
    parser.add_argument(
        "--default", default="5.00#5.00", help="若無法解析 V#A 時使用的備援值 (預設 5.00#5.00)"
    )
    return parser.parse_args()


def format_va(value: str, fallback: str) -> str:
    value = value.strip()
    if "#" not in value:
        return fallback
    valence_str, arousal_str = value.split("#", 1)
    valence_str = valence_str.strip()
    arousal_str = arousal_str.strip()
    try:
        valence = float(valence_str)
        arousal = float(arousal_str)
        return f"{valence:.2f}#{arousal:.2f}"
    except ValueError:
        return fallback


def process_file(input_path: str, output_path: str, fallback: str) -> None:
    rows = load_jsonl(input_path)
    for record in rows:
        quads = record.get("Quadruplet", [])
        for quad in quads:
            va = quad.get("VA")
            if isinstance(va, str):
                quad["VA"] = format_va(va, fallback)
    save_jsonl(output_path, rows)


def main() -> None:
    args = parse_args()
    process_file(args.input_path, args.output_path, args.default)


if __name__ == "__main__":
    main()

