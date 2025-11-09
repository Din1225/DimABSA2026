"""
使用多個 LLM 預測結果進行 VA 平均集成。
"""

from __future__ import annotations

import argparse
from typing import Sequence

from .data_utils import load_jsonl, save_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average VA predictions from multiple JSONL files.")
    parser.add_argument(
        "--input-paths",
        nargs="+",
        required=True,
        help="多個來源 JSONL 檔案（格式需與 predict_task3 輸出一致）",
    )
    parser.add_argument("--output-path", required=True, help="輸出的平均後 JSONL 檔案")
    parser.add_argument(
        "--fallback-va",
        default="5.00#5.00",
        help="若某檔案無法解析 VA 時使用的備援值（預設 5.00#5.00）",
    )
    return parser.parse_args()


def parse_va(value: str) -> tuple[float, float] | None:
    text = value.strip()
    if "#" not in text:
        return None
    left, right = text.split("#", 1)
    try:
        return float(left), float(right)
    except ValueError:
        return None


def format_va(valence: float, arousal: float) -> str:
    return f"{valence:.2f}#{arousal:.2f}"


def ensure_alignment(rows_list: Sequence[list[dict]], input_paths: Sequence[str]) -> None:
    reference_len = len(rows_list[0])
    for path, rows in zip(input_paths, rows_list):
        if len(rows) != reference_len:
            raise ValueError(f"檔案 {path} 的筆數 ({len(rows)}) 與基準 ({reference_len}) 不符。")


def average_record(
    records: Sequence[dict],
    fallback_va: tuple[float, float],
) -> dict:
    # 以第一個檔案為模板，僅替換 Quadruplet 內的 VA。
    base = dict(records[0])
    quad_lists = [rec.get("Quadruplet", []) or [] for rec in records]
    reference_quads = quad_lists[0]
    for path_idx, quads in enumerate(quad_lists[1:], start=1):
        if len(quads) != len(reference_quads):
            raise ValueError(
                f"ID {base.get('ID')} 在第 {path_idx + 1} 個檔案的 quadruplet 數量 "
                f"{len(quads)} 與基準 {len(reference_quads)} 不一致。"
            )
    averaged_quads: list[dict] = []
    for q_idx, template in enumerate(reference_quads):
        values: list[tuple[float, float]] = []
        for quads in quad_lists:
            quad = quads[q_idx]
            for key in ("Aspect", "Category", "Opinion"):
                if quad.get(key) != template.get(key):
                    raise ValueError(
                        f"ID {base.get('ID')} 的第 {q_idx} 筆 {key} 不一致：{quad.get(key)} vs {template.get(key)}."
                    )
            parsed = parse_va(str(quad.get("VA", "")).strip())
            if parsed is None:
                parsed = fallback_va
            values.append(parsed)
        valence_avg = sum(v for v, _ in values) / len(values)
        arousal_avg = sum(a for _, a in values) / len(values)
        new_quad = dict(template)
        new_quad["VA"] = format_va(valence_avg, arousal_avg)
        averaged_quads.append(new_quad)
    base["Quadruplet"] = averaged_quads
    return base


def compute_ensemble(input_paths: Sequence[str], fallback_va: tuple[float, float]) -> list[dict]:
    rows_list = [load_jsonl(path) for path in input_paths]
    ensure_alignment(rows_list, input_paths)
    output_rows: list[dict] = []
    reference_rows = rows_list[0]
    for idx in range(len(reference_rows)):
        ref_id = reference_rows[idx].get("ID")
        aligned_records: list[dict] = []
        for path, rows in zip(input_paths, rows_list):
            record = rows[idx]
            if record.get("ID") != ref_id:
                raise ValueError(
                    f"第 {idx} 筆的 ID 不一致：{ref_id} vs {record.get('ID')} (來源 {path})。"
                )
            aligned_records.append(record)
        output_rows.append(average_record(aligned_records, fallback_va))
    return output_rows


def main() -> None:
    args = parse_args()
    fallback = parse_va(args.fallback_va)
    if fallback is None:
        raise ValueError(f"無法解析 fallback VA: {args.fallback_va}")
    rows = compute_ensemble(args.input_paths, fallback)
    save_jsonl(args.output_path, rows)


if __name__ == "__main__":
    main()
