# CSE234 Project 2 — Schema Linking with SFT
**Due: Mon Jun 1, 11:59pm PT**

---

## Task Summary

Given a natural language question + a database ID, output which tables and columns the
underlying SQL would reference, as structured JSON:

```json
{"TableName1": ["Col1", "Col2"], "TableName2": []}
```

Graded on table-level and column-level Precision/Recall/F1 (macro-averaged).  
Leaderboard score = `0.5 * (P+R+F1)/3 tables + 0.5 * (P+R+F1)/3 columns`. Target: >= 0.60.

---

## Hard Constraints

- Base model <= 2B parameters, no external APIs at inference
- Must run on single DSMLP GPU (~24 GB, NVIDIA RTX PRO 6000 Blackwell)
- Inference for ~100 questions within 15 minutes
- LoRA adapter committed to `adapter/` in repo (~10-100 MB)
- `logs/` must contain real RapidFire AI logs matching report claims (TA spot-check)
- Git history must show genuine iterative development (no dump commits)

---

## Repo Structure (target)

```
CSE234_proj2/
├── schemas/                  # 17 Spider-format JSON schemas — committed
├── adapter/                  # final LoRA adapter weights — committed
├── logs/                     # RapidFire AI experiment logs — committed
├── data/
│   ├── train.json
│   ├── validation.json
│   ├── validation_input.json
│   ├── validation_gold_schema_links.json
│   └── augmented/            # optional generated data
├── main.py                   # inference entry point — graded
├── train.py                  # SFT training script
├── data_prep.py              # prompt formatting + augmentation helpers
├── eval.py                   # provided grader — do not modify
├── sql_to_schema_links.py    # provided helper — do not modify
├── sample_main.py            # provided stub — do not modify
├── README.md
└── report.pdf
```

---

## Base Model Choice

| Model | Params | Notes |
|---|---|---|
| Qwen2.5-1.5B-Instruct | 1.5B | Primary choice — strong JSON output |
| Qwen3-1.7B | 1.7B | Newer; set `enable_thinking=False` |
| SmolLM2-1.7B-Instruct | 1.7B | Very efficient |
| Llama-3.2-1B-Instruct | 1B | Good smaller ablation baseline |

**Start with Qwen2.5-1.5B-Instruct.**

---

## Prompt Template (v1 baseline)

```
System:
You are a schema linking assistant. Given a natural language question and a database
schema, output a JSON object mapping each referenced table name to a list of referenced
column names. Use an empty list for tables referenced without specific columns (e.g.
COUNT(*)). Output only valid JSON — no explanation.

User:
Database: {db_id}
Schema:
{schema_serialization}

Question: {question}