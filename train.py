import argparse
import json
import os

from datasets import Dataset
from rapidfireai import Experiment
from rapidfireai.automl import List, RFGridSearch, RFModelConfig, RFLoraConfig, RFSFTConfig

from data_prep import build_dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EXPERIMENTS_PATH = "./logs"

# 8 configs required by the report — vary model, LoRA rank, lr, target modules
# Structured as a grid: 2 models x 2 LoRA ranks x 2 LRs = 8 configs

MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
]

LORA_CONFIGS = List([
    RFLoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    ),
    RFLoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    ),
])

LEARNING_RATES = [2e-4, 5e-5]


# ---------------------------------------------------------------------------
# Data formatting for RapidFire AI
# ---------------------------------------------------------------------------

def formatting_func(row):
    messages = row["messages"]
    # split into prompt (system + user) and completion (assistant)
    prompt = [m for m in messages if m["role"] in ("system", "user")]
    completion = [m for m in messages if m["role"] == "assistant"]
    return {"prompt": prompt, "completion": completion}


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------

def create_model(model_config):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    name = model_config["model_name"]
    kwargs = model_config["model_kwargs"]
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    tokenizer.padding_side = "right"
    return model, tokenizer


# ---------------------------------------------------------------------------
# Build config grid
# ---------------------------------------------------------------------------

def build_config_group(args):
    model_configs = []
    for model_name in MODELS:
        for lr in LEARNING_RATES:
            model_configs.append(
                RFModelConfig(
                    model_name=model_name,
                    peft_config=LORA_CONFIGS,
                    training_args=RFSFTConfig(
                        learning_rate=lr,
                        lr_scheduler_type="cosine",
                        per_device_train_batch_size=args.batch_size,
                        per_device_eval_batch_size=args.batch_size,
                        num_train_epochs=args.epochs,
                        gradient_accumulation_steps=args.grad_accum,
                        logging_steps=10,
                        eval_strategy="epoch",
                        save_strategy="epoch",
                        bf16=True,
                        max_length=args.max_seq_len,
                        gradient_checkpointing=True,
                        output_dir=args.output_dir,
                    ),
                    model_type="causal_lm",
                    model_kwargs={
                        "device_map": "auto",
                        "dtype": "auto",
                        "use_cache": False,
                        "trust_remote_code": True,
                    },
                    formatting_func=formatting_func,
                )
            )

    return RFGridSearch(configs=List(model_configs), trainer_type="SFT")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="Project2/train.json")
    ap.add_argument("--validation", default="Project2/validation.json")
    ap.add_argument("--schemas_dir", default="Project2/schemas")
    ap.add_argument("--output_dir", default="./runs")
    ap.add_argument("--experiment_name", default="schema-linking-v1")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--num_chunks", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(EXPERIMENTS_PATH, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Building datasets...")
    train_examples = build_dataset(args.train, args.schemas_dir)
    val_examples = build_dataset(args.validation, args.schemas_dir)
    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)
    print(f"  train: {len(train_dataset)}, val: {len(val_dataset)}")

    print(f"Starting experiment: {args.experiment_name}")
    experiment = Experiment(
        experiment_name=args.experiment_name,
        mode="fit",
        experiment_path=EXPERIMENTS_PATH,
    )

    config_group = build_config_group(args)

    experiment.run_fit(
        config_group,
        create_model,
        train_dataset,
        val_dataset,
        num_chunks=args.num_chunks,
        seed=42,
    )

    experiment.end()
    print("Done. Logs written to", EXPERIMENTS_PATH)
    print("Copy your best adapter to ./adapter/ before running main.py")


if __name__ == "__main__":
    main()
