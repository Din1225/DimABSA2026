import argparse
import json
from typing import Dict, List, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_PROMPT = (
    "你是一個情感計量模型。針對輸入的文本、面向與主觀詞，請預測對應的 Valence (V) 與 Arousal (A) 分數，"
    "並以 JSON 格式輸出 \"va\" 欄位，其值必須是 V#A 並保留兩位小數，介於 1.00 到 9.00 之間。"
)


def load_model(model_id: str, adapter_path: str, cache_dir: str):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        use_safetensors=True,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto",
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def predict_va(
    text: str,
    aspect: str,
    opinion: str,
    model,
    tokenizer,
    max_new_tokens: int,
) -> Tuple[str, str]:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"{SYSTEM_PROMPT}\n"
                        f"文本：{text}\n"
                        f"面向：{aspect}\n"
                        f"主觀詞：{opinion}"
                    ),
                }
            ],
        }
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    text_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    try:
        parsed = json.loads(text_output)
        value = parsed.get("va")
        if isinstance(value, str):
            return value, text_output
    except json.JSONDecodeError:
        pass
    return "5.00#5.00", text_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="JSONL，需包含 Text 與 Predicted 欄位")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model_id", default="google/gemma-3-4b-it")
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--cache_dir", default="/workplace/Share/LLM_model")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_id, args.adapter_path or None, args.cache_dir)

    results: List[Dict[str, object]] = []
    with open(args.input_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            text = record["Text"]
            enriched = []
            for item in record.get("Predicted", []):
                va, raw_text = predict_va(
                    text,
                    item["Aspect"],
                    item["Opinion"],
                    model,
                    tokenizer,
                    args.max_new_tokens,
                )
                enriched.append({
                    "Aspect": item["Aspect"],
                    "Opinion": item["Opinion"],
                    "Category": item.get("Category", ""),
                    "VA": va,
                    "raw_generation": raw_text,
                })
            results.append({"ID": record["ID"], "Text": text, "Quadruplet": enriched})

    with open(args.output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
