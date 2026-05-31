# CSE234 Project 2 — Schema Linking with SFT

Schema linking for NL-to-SQL: given a natural language question and a database schema, identify which tables and columns the underlying SQL would reference.

## Repo structure

```
CSE234_proj2/
├── Project2/           starter code, data, schemas (from release packet)
│   ├── schemas/        17 Spider-format DB schemas
│   ├── train.json      301 training examples
│   ├── validation.json / validation_input.json / validation_gold_schema_links.json
│   ├── eval.py         grader (do not modify)
│   └── sql_to_schema_links.py
├── adapter/            trained LoRA adapter weights (committed after training)
├── logs/               RapidFire AI experiment logs
├── main.py             inference entry point (graded)
├── train.py            SFT training with RapidFire AI
├── data_prep.py        schema serialization + prompt formatting
└── README.md
```

## Setup

```bash
pip install rapidfireai transformers peft trl datasets torch accelerate
```

On DSMLP the base environment already has PyTorch — only install what is missing.

## Training (run on DSMLP)

```bash
cd /path/to/CSE234_proj2
python train.py \
    --train      Project2/train.json \
    --validation Project2/validation.json \
    --schemas_dir Project2/schemas \
    --experiment_name schema-linking-v1 \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 4 \
    --max_seq_len 2048
```

Logs are written to `./logs/`. After training, copy the best adapter checkpoint to `./adapter/`:

```bash
cp -r runs/<best_run>/adapter_checkpoint/ adapter/
```

## Inference (graded command)

```bash
python main.py \
    --input  Project2/validation_input.json \
    --output preds.json
```

Schemas are loaded from `./schemas/` by default. Symlink or copy `Project2/schemas` to the repo root if running from repo root:

```bash
ln -s Project2/schemas schemas
```

Optional flags:
- `--schemas_dir <path>` — override schema directory (default: `./schemas`)
- `--adapter_path <path>` — override adapter path (default: `./adapter`)
- `--batch_size <int>` — inference batch size (default: 8)

## Evaluate on validation set

```bash
python Project2/eval.py \
    --predictions preds.json \
    --gold        Project2/validation_gold_schema_links.json \
    --schemas_dir Project2/schemas \
    --questions_input Project2/validation_input.json \
    --per_question_out per_q.csv
```

## Model

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct` (loaded from HuggingFace Hub at inference)
- **Adapter:** LoRA weights in `./adapter/` (committed to this repo)
- **Loading:** `PeftModel.from_pretrained(base, "./adapter")`
