import argparse
import json
import os
from typing import List, Dict

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from trl import SFTConfig, SFTTrainer

print(torch.__version__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", DEVICE)

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")

SYSTEM_PROMPT = (
    "你是一個中文情感分析模型。請從提供的文本中抽取所有的面向詞 (Aspect) 與對應的主觀詞 (Opinion)，"
    "並以 JSON 陣列形式輸出，每個元素包含兩個欄位：aspect 與 opinion。"
    "若無任何面向，請輸出空陣列。"
)


def load_training_samples(path: str) -> List[Dict[str, object]]:
    examples: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            quadruplets = record.get("Quadruplet", [])
            target = [
                {"aspect": item["Aspect"], "opinion": item["Opinion"]}
                for item in quadruplets
            ]
            examples.append({"text": record["Text"], "target": json.dumps(target, ensure_ascii=False)})
    return examples


def create_prompt(sample: Dict[str, str], tokenizer) -> Dict[str, str]:
    user_msg = f"{SYSTEM_PROMPT}\n文本：{sample['text']}"
    assistant_msg = sample["target"]
    messages = [
        {"role": "user", "content": [{"type": "text", "text": user_msg}]},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_msg}]},
    ]
    sample["prompt"] = tokenizer.apply_chat_template(messages, tokenize=False)
    return sample


def preprocess_dataset(dataset: Dataset, tokenizer, max_seq_len: int) -> Dataset:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = max_seq_len

    formatted = dataset.map(lambda sample: create_prompt(sample, tokenizer))
    tokenized = formatted.map(
        lambda batch: tokenizer(batch["prompt"], max_length=max_seq_len, truncation=True, padding="max_length"),
        batched=True,
        remove_columns=formatted.column_names,
    )
    return tokenized


def build_trainer(model, dataset: Dataset, tokenizer, args) -> SFTTrainer:
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        optim="paged_adamw_32bit",
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        bf16=True,
        fp16=False,
        max_grad_norm=0.3,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=0.05,
        save_strategy="epoch",
        group_by_length=True,
        output_dir=args.output_dir,
        save_safetensors=True,
        lr_scheduler_type="cosine",
        seed=42,
        packing=False,
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        args=sft_config,
        data_collator=data_collator,
    )
    return trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="trial_data/_train.jsonl")
    parser.add_argument("--model_id", default="google/gemma-3-4b-it")
    parser.add_argument("--output_dir", default="./checkpoints/aspect_opinion")
    parser.add_argument("--cache_dir", default="/workplace/Share/LLM_model")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    offload_dir = os.path.join(args.output_dir, "offload")
    os.makedirs(offload_dir, exist_ok=True)

    print(f"MODEL_ID = {args.model_id}")
    print(f"OUTPUT_DIR = {os.path.abspath(args.output_dir)}")
    print(f"CACHE_DIR = {args.cache_dir}")

    if torch.cuda.is_available():
        free_gb = {i: torch.cuda.mem_get_info(i)[0] // (1024 ** 3) for i in range(torch.cuda.device_count())}
        max_memory = {i: f"{max(free_gb[i] - 2, 1)}GiB" for i in free_gb}
        max_memory["cpu"] = "48GiB"
    else:
        max_memory = None

    records = load_training_samples(args.train_path)
    dataset = Dataset.from_list(records)
    print(f"Loaded {len(dataset)} training samples")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    if not hasattr(tokenizer, "apply_chat_template"):
        raise AttributeError("所選模型的 tokenizer 不支援 chat template，請改用具備對話模版的指令模型。")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        use_safetensors=True,
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        device_map="auto",
        max_memory=max_memory,
        offload_folder=offload_dir,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    tokenized = preprocess_dataset(dataset, tokenizer, args.max_seq_len)
    trainer = build_trainer(model, tokenized, tokenizer, args)

    train_result = trainer.train()
    trainer.save_state()

    adapter_dir = os.path.join(args.output_dir, "adapter")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    with open(os.path.join(args.output_dir, "train_result.json"), "w", encoding="utf-8") as f:
        json.dump({"train_loss": train_result.training_loss}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
