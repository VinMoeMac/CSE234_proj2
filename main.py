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

COLUMN_SYSTEM_PROMPT = (
    "You are a column linking assistant. Given a natural language question and a single "
    "database table with its columns, output a JSON list of column names that the question "
    "references. If no specific columns are referenced (e.g. COUNT(*) or SELECT *), output "
    "an empty list []. Output only a valid JSON array — no explanation, no extra text."
)


def load_model(adapter_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path, local_files_only=True)
    model = model.to(torch.bfloat16)
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


def keyword_fallback(question: str, schema: dict) -> dict:
    tables = schema["table_names_original"]
    columns = schema["column_names_original"]
    q_tokens = set(re.sub(r"[^a-z0-9]", " ", question.lower()).split())

    # score each table by token overlap with question
    scores = {}
    for t in tables:
        t_tokens = set(re.sub(r"[^a-z0-9]", " ", t.lower()).split())
        scores[t] = len(t_tokens & q_tokens)

    best_score = max(scores.values())
    if best_score > 0:
        matched = [t for t, s in scores.items() if s == best_score]
    else:
        # no match — return top 2 by score to improve recall
        sorted_tables = sorted(scores, key=scores.get, reverse=True)
        matched = sorted_tables[:2]

    table_to_cols = {}
    for tbl_idx, col_name in columns:
        if tbl_idx == -1:
            continue
        t = tables[tbl_idx]
        table_to_cols.setdefault(t, []).append(col_name)

    result = {}
    for t in matched:
        col_matches = [
            c for c in table_to_cols.get(t, [])
            if set(re.sub(r"[^a-z0-9]", " ", c.lower()).split()) & q_tokens
        ]
        result[t] = col_matches
    return result


def augment_columns(question: str, links: dict, schema: dict) -> dict:
    """For tables the model predicted, add keyword-matched columns it may have missed."""
    columns = schema["column_names_original"]
    tables = schema["table_names_original"]
    q_tokens = set(re.sub(r"[^a-z0-9]", " ", question.lower()).split())

    table_to_cols = {}
    for tbl_idx, col_name in columns:
        if tbl_idx == -1:
            continue
        t = tables[tbl_idx]
        table_to_cols.setdefault(t.lower(), []).append(col_name)

    result = {}
    for table, pred_cols in links.items():
        extra = [
            c for c in table_to_cols.get(table.lower(), [])
            if set(re.sub(r"[^a-z0-9]", " ", c.lower()).split()) & q_tokens
            and c not in pred_cols
        ]
        result[table] = pred_cols + extra
    return result


def truncate_schema_text(schema_text: str, question: str, tokenizer, max_schema_tokens: int) -> str:
    """Truncate schema lines from the bottom using binary search."""
    lines = schema_text.split("\n")
    if not lines:
        return ""
    # fast check: does the full text fit?
    full_tokens = len(tokenizer.encode("\n".join(lines)))
    if full_tokens <= max_schema_tokens:
        return schema_text
    # binary search on number of lines to keep
    lo, hi = 1, len(lines)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = "\n".join(lines[:mid])
        tokens = len(tokenizer.encode(candidate))
        if tokens <= max_schema_tokens:
            lo = mid
        else:
            hi = mid - 1
    return "\n".join(lines[:lo])


def run_batch(model, tokenizer, prompts: list[str], max_seq_len: int = 3072) -> list[str]:
    # Reserve tokens: system prompt (~200) + question (~100) + generation (~512) + overhead (~200)
    max_schema_tokens = max_seq_len - 512 - 200

    # Truncate schema portion of each prompt to protect system prompt and question
    truncated_prompts = []
    for p in prompts:
        # prompt format: "Database: X\nSchema:\n<schema>\n\nQuestion: Y"
        if "\n\nQuestion:" in p:
            schema_part, question_part = p.split("\n\nQuestion:", 1)
            if "\nSchema:\n" in schema_part:
                db_part, schema_text = schema_part.split("\nSchema:\n", 1)
                schema_text = truncate_schema_text(schema_text, question_part, tokenizer, max_schema_tokens)
                p = f"{db_part}\nSchema:\n{schema_text}\n\nQuestion:{question_part}"
        truncated_prompts.append(p)

    chats = [
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}]
        for p in truncated_prompts
    ]
    encoded = tokenizer.apply_chat_template(
        chats,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        encoded,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_len,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    results = []
    for i, out in enumerate(outputs):
        generated = out[inputs["input_ids"].shape[1]:]
        results.append(tokenizer.decode(generated, skip_special_tokens=True))
    return results


def stage2_columns(
    model, tokenizer, question: str, table: str, col_names: list[str],
    max_seq_len: int = 2048
) -> list[str]:
    """Stage 2: given predicted table, ask model which columns are referenced."""
    col_list = "\n".join(f"  - {c}" for c in col_names)
    prompt = (
        f"Question: {question}\n\n"
        f"Table: {table}\n"
        f"Columns:\n{col_list}\n\n"
        f"Which columns from this table does the question reference? "
        f"Output a JSON array of column names, or [] if none."
    )
    chat = [{"role": "system", "content": COLUMN_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template([chat], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(encoded, return_tensors="pt", truncation=True,
                       max_length=max_seq_len).to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # parse JSON array
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return [c for c in result if isinstance(c, str)]
    except json.JSONDecodeError:
        pass
    # fallback: find array in output
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, list):
                return [c for c in result if isinstance(c, str)]
        except json.JSONDecodeError:
            pass
    return []


def predict_all(
    questions: list[dict],
    schemas_dir: str,
    model,
    tokenizer,
    batch_size: int = BATCH_SIZE,
    filter_schema: bool = False,
    max_seq_len: int = 3072,
    two_stage: bool = False,
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
                filter_question=q["question"],
                show_fk_links=True,
            )
            prompts.append(format_prompt(q["question"], q["db_id"], schema_text))

        raw_outputs = run_batch(model, tokenizer, prompts, max_seq_len=max_seq_len)

        for q, schema, raw in zip(batch, schemas, raw_outputs):
            links = extract_json(raw) or {}
            links = validate_links(links, schema)
            if not links:
                links = keyword_fallback(q["question"], schema)

            if two_stage and links:
                # Stage 2: refine columns for each predicted table
                table_col_map = {}
                for tbl_idx, col_name in schema["column_names_original"]:
                    if tbl_idx == -1:
                        continue
                    t = schema["table_names_original"][tbl_idx]
                    table_col_map.setdefault(t, []).append(col_name)

                refined = {}
                for table in links:
                    all_cols = table_col_map.get(table, [])
                    if not all_cols:
                        refined[table] = []
                        continue
                    predicted = stage2_columns(model, tokenizer, q["question"],
                                               table, all_cols, max_seq_len=max_seq_len)
                    # validate predicted columns against schema
                    col_lower = {c.lower(): c for c in all_cols}
                    valid = [col_lower[c.lower()] for c in predicted
                             if c.lower() in col_lower]
                    refined[table] = valid if valid else links[table]  # fallback to stage1
                links = refined

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
    ap.add_argument("--two_stage", action="store_true",
                    help="Two-stage inference: predict tables first, then refine columns per table")
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
        two_stage=args.two_stage,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(preds)} predictions to {args.output}")


if __name__ == "__main__":
    main()
