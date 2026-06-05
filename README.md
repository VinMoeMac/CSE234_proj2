# CSE234 Project 2 — Schema Linking with SFT

Schema linking for NL-to-SQL: given a natural language question and a database schema, identify which tables and columns the underlying SQL would reference.

## Quick Start (Inference Only)

TAs: clone the repo, install dependencies, and run inference. No training needed — the trained adapter is committed at `adapter/`.

```bash
pip install transformers peft torch accelerate
python3 main.py --input <input.json> --output <predictions.json>
```

The graded command works out of the box:
- Schemas load from `./Project2/schemas/` by default
- Adapter loads from `./adapter/` by default
- Base model (`Qwen/Qwen2.5-1.5B-Instruct`) downloads automatically from HuggingFace Hub
- Two-stage inference is enabled by default

## Dependencies

```bash
pip install transformers peft torch accelerate datasets trl
```

## Inference

```bash
python3 main.py --input <questions.json> --output <predictions.json>
```

Optional overrides:
- `--schemas_dir <path>` — schema directory (default: `./Project2/schemas`)
- `--adapter_path <path>` — adapter directory (default: `./adapter`)
- `--batch_size <int>` — inference batch size (default: 4)

Expected runtime: ~3-4 minutes for 100 questions on a 24GB GPU.

## Evaluate

```bash
python3 Project2/eval.py \
    --predictions <predictions.json> \
    --gold        Project2/validation_gold_schema_links.json \
    --schemas_dir Project2/schemas \
    --questions_input Project2/validation_input.json
```

## Model Details

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct` (1.5B params, auto-downloaded from HuggingFace)
- **Adapter:** LoRA r=32, committed at `adapter/` (~34MB)
- **Validation score:** 0.4539 leaderboard (Table 0.517, Column 0.394)

## Repo Structure

```
CSE234_proj2/
├── adapter/            LoRA adapter weights (committed)
├── Project2/           Starter code + data + schemas
│   ├── schemas/        17 Spider-format DB schemas
│   ├── train.json      301 training examples
│   ├── eval.py         grader
│   └── ...
├── augmented_data/     Synthetic training examples (SAP + NTSB)
├── logs/               Experiment logs
├── main.py             Inference entry point
├── train_simple.py     Training script (TRL SFTTrainer)
├── data_prep.py        Schema serialization
├── report.pdf          Project report
└── README.md
```

## Training (not needed for grading)

```bash
python train_simple.py \
    --train augmented_data/train_v2.json \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --epochs 6 --batch_size 1 --grad_accum 16 \
    --lr 8e-5 --max_seq_len 2048 \
    --show_fk_links --sort_by_question --lora_r 32 \
    --output_dir ./runs/my_run
```
