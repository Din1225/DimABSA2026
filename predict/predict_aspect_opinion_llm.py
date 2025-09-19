import argparse
import json
from typing import List, Dict, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_PROMPT = (
    "你是一個中文情感分析模型。請從提供的文本中抽取所有的面向詞 (Aspect) 與對應的主觀詞 (Opinion)，"
    "並以 JSON 陣列形式輸出，每個元素包含兩個欄位：aspect 與 opinion。"
    "若無任何面向，請輸出空陣列。"
)


def build_model(model_id: str, adapter_path: str, cache_dir: str):
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


def extract_pairs(
    text: str,
    model,
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 256,
) -> Tuple[List[Dict[str, str]], str]:
    messages = [
        {"role": "user", "content": [{"type": "text", "text": f"{SYSTEM_PROMPT}\n文本：{text}"}]}
    ]
    chat_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
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
        if isinstance(parsed, list):
            cleaned = []
            for item in parsed:
                aspect = item.get("aspect")
                opinion = item.get("opinion")
                if isinstance(aspect, str) and isinstance(opinion, str):
                    cleaned.append({"Aspect": aspect, "Opinion": opinion})
            return cleaned, text_output
    except json.JSONDecodeError:
        pass
    return [], text_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="JSONL 檔案，包含 ID 與 Text 欄位")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model_id", default="google/gemma-3-4b-it")
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--cache_dir", default="/workplace/Share/LLM_model")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = build_model(args.model_id, args.adapter_path or None, args.cache_dir)

    results = []
    with open(args.input_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            pairs, raw_text = extract_pairs(
                record["Text"],
                model,
                tokenizer,
                device,
                args.max_new_tokens,
            )
            results.append(
                {
                    "ID": record["ID"],
                    "AspectOpinion": {
                        "raw_generation": raw_text,
                        "pairs": pairs,
                    },
                }
            )

    with open(args.output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
