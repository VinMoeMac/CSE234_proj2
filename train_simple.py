"""
Baseline training script — plain TRL SFTTrainer, no RapidFire dependency.
Use this for local dev/testing. For the real multi-config runs use train.py on DSMLP.
"""
import argparse
import os

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

import json

from data_prep import build_dataset

# Databases that score poorly — oversample these
UNDERSAMPLE_DBS = {
    "NTSB",
    "SBODemoUS-Reports", "SBODemoUS-General", "SBODemoUS-Finance",
    "SBODemoUS-Service", "SBODemoUS-Human Resources", "SBODemoUS-Business Partners",
    "SBODemoUS-Inventory and Production", "SBODemoUS-Banking", "SBODemoUS-Sales Opportunities",
}


def oversample(data: list[dict], train_json: str, factor: int = 4) -> list[dict]:
    """Repeat examples from underrepresented databases by factor."""
    with open(train_json) as f:
        raw = json.load(f)
    db_by_qid = {ex["question_id"]: ex["db_id"] for ex in raw}

    # data_prep returns messages dicts without db_id — re-attach from raw
    paired = list(zip(raw, data))  # assumes same order, same length
    result = list(data)
    for ex_raw, ex_data in paired:
        if ex_raw["db_id"] in UNDERSAMPLE_DBS:
            result.extend([ex_data] * (factor - 1))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="Project2/train.json")
    ap.add_argument("--validation", default="Project2/validation.json")
    ap.add_argument("--schemas_dir", default="Project2/schemas")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--output_dir", default="./runs/baseline")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--filter_schema", action="store_true")
    ap.add_argument("--oversample_factor", type=int, default=4,
                    help="How many times to repeat underrepresented DB examples")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    print("Building datasets...")
    train_data = build_dataset(args.train, args.schemas_dir, filter_schema=args.filter_schema)
    val_data = build_dataset(args.validation, args.schemas_dir, filter_schema=args.filter_schema)

    if args.oversample_factor > 1:
        train_data = oversample(train_data, args.train, factor=args.oversample_factor)
        print(f"  Oversampled underrepresented DBs by {args.oversample_factor}x")

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    print(f"  train: {len(train_dataset)}, val: {len(val_dataset)}")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        max_length=args.max_seq_len,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("Training...")
    trainer.train()

    adapter_path = os.path.join(args.output_dir, "adapter")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Adapter saved to {adapter_path}")
    print("Done. Copy adapter/ to repo root before running main.py")


if __name__ == "__main__":
    main()
