"""
VA regression prompt：統一的 instruct 風格模板。
"""

from __future__ import annotations


SYSTEM_PROMPT = (
    "你是一個 Valence-Arousal 強度預測助手。"
    "收到 Text、Aspect 與 Opinion 後，只能輸出一行 `V#A`。"
    "規則：1) Valence 在前、Arousal 在後，兩者之間用 `#`，"
    "2) 數值範圍 1.0 至 9.0，請輸出整數，"
    "3) 不得輸出多餘文字、標點或解釋。"
)


def build_instruct_prompt(text: str, aspect: str, opinion: str) -> str:
    """
    建立 instruct 風格 prompt，供訓練與推論共用。
    """
    safe_text = text.replace("\n", " ").strip()
    safe_aspect = aspect.replace("\n", " ").strip()
    safe_opinion = opinion.replace("\n", " ").strip()
    return (
        f"{SYSTEM_PROMPT}\n"
        f"Text: {safe_text}\n"
        f"Aspect: {safe_aspect}\n"
        f"Opinion: {safe_opinion}\n"
        "請輸出："
    )
