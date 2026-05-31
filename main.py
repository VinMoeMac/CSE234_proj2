import argparse
import json
import os
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_prep import load_schema, serialize_schema, format_prompt, SYSTEM_PROMPT

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "./adapter"
DEFAULT_SCHEMAS_DIR = "./schemas"
MAX_NEW_TOKENS = 512
BATCH_SIZE = 4


def load_model(adapter_path: str):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


def extract_json(text: str) -> dict | None:
    text = text.strip()
    # try direct parse first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # find first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def validate_links(links: dict, schema: dict) -> dict:
    tables = schema["table_names_original"]
    columns = schema["column_names_original"]  # [tbl_idx, col_name]

    # case-insensitive table lookup
    table_lower = {t.lower(): t for t in tables}

    # per-table column lookup: table_lower_name -> {col_lower: col_original}
    table_cols = {}
    for tbl_idx, col_name in columns:
        if tbl_idx == -1:
            continue
        t_lower = tables[tbl_idx].lower()
        table_cols.setdefault(t_lower, {})[col_name.lower()] = col_name

    validated = {}
    for pred_table, pred_cols in links.items():
        if not isinstance(pred_cols, list):
            continue
        canon_table = table_lower.get(pred_table.lower())
        if canon_table is None:
            continue  # hallucinated table — drop it
        col_map = table_cols.get(canon_table.lower(), {})
        valid_cols = []
        seen = set()
        for col in pred_cols:
            if not isinstance(col, str):
                continue
            canon_col = col_map.get(col.lower())
            if canon_col and canon_col.lower() not in seen:
                valid_cols.append(canon_col)
                seen.add(canon_col.lower())
            # hallucinated column — drop it
        validated[canon_table] = valid_cols
    return validated


def run_batch(model, tokenizer, prompts: list[str], max_seq_len: int = 3072) -> list[str]:
    chats = [
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}]
        for p in prompts
    ]
    inputs = tokenizer.apply_chat_template(
        chats,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_len,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    results = []
    for out in outputs:
        generated = out[inputs.shape[1]:]
        results.append(tokenizer.decode(generated, skip_special_tokens=True))
    return results


def predict_all(
    questions: list[dict],
    schemas_dir: str,
    model,
    tokenizer,
    batch_size: int = BATCH_SIZE,
    filter_schema: bool = False,
    max_seq_len: int = 3072,
) -> list[dict]:
    preds = []
    total = len(questions)
    for i in range(0, total, batch_size):
        batch = questions[i: i + batch_size]
        prompts = []
        schemas = []
        for q in batch:
            schema = load_schema(q["db_id"], schemas_dir)
            schemas.append(schema)
            schema_text = serialize_schema(
                schema,
                filter_question=q["question"] if filter_schema else None,
            )
            prompts.append(format_prompt(q["question"], q["db_id"], schema_text))

        raw_outputs = run_batch(model, tokenizer, prompts, max_seq_len=max_seq_len)

        for q, schema, raw in zip(batch, schemas, raw_outputs):
            links = extract_json(raw) or {}
            links = validate_links(links, schema)
            preds.append({"question_id": q["question_id"], "schema_links": links})

        done = min(i + batch_size, total)
        print(f"  {done}/{total} done")
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--schemas_dir", default=DEFAULT_SCHEMAS_DIR)
    ap.add_argument("--adapter_path", default=ADAPTER_PATH)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--max_seq_len", type=int, default=3072)
    ap.add_argument("--filter_schema", action="store_true",
                    help="Only serialize tables matching question tokens (use on low VRAM)")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Loading model {BASE_MODEL} + adapter {args.adapter_path}...")
    model, tokenizer = load_model(args.adapter_path)

    print(f"Running inference on {len(questions)} questions...")
    preds = predict_all(
        questions,
        args.schemas_dir,
        model,
        tokenizer,
        batch_size=args.batch_size,
        filter_schema=args.filter_schema,
        max_seq_len=args.max_seq_len,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(preds)} predictions to {args.output}")


if __name__ == "__main__":
    main()
