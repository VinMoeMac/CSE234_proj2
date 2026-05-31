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
BATCH_SIZE = 8


def load_model(adapter_path: str):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


def extract_json(text: str) -> str | None:
    text = text.strip()
    # try to find a JSON object in the output
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return None


def validate_links(links: dict, schema: dict) -> dict:
    tables = {t.lower(): t for t in schema["table_names_original"]}
    columns = schema["column_names_original"]  # [tbl_idx, col_name]

    table_to_cols = {}
    for tbl_idx, col_name in columns:
        if tbl_idx == -1:
            continue
        t = schema["table_names_original"][tbl_idx]
        table_to_cols.setdefault(t.lower(), set()).add(col_name.lower())

    validated = {}
    for pred_table, pred_cols in links.items():
        canon_table = tables.get(pred_table.lower())
        if canon_table is None:
            continue  # hallucinated table — drop it
        valid_cols = []
        for col in pred_cols:
            if col.lower() in table_to_cols.get(canon_table.lower(), set()):
                # find original casing
                for _, c in columns:
                    if c.lower() == col.lower():
                        valid_cols.append(c)
                        break
            # hallucinated column — drop it
        validated[canon_table] = valid_cols
    return validated


def run_batch(model, tokenizer, prompts: list[str]) -> list[str]:
    inputs = tokenizer.apply_chat_template(
        [[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}]
         for p in prompts],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    results = []
    for i, out in enumerate(outputs):
        generated = out[inputs.shape[1]:]
        results.append(tokenizer.decode(generated, skip_special_tokens=True))
    return results


def predict_all(questions: list[dict], schemas_dir: str, model, tokenizer, batch_size: int = BATCH_SIZE, filter_schema: bool = False) -> list[dict]:
    preds = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i: i + batch_size]
        prompts = []
        schemas = []
        for q in batch:
            schema = load_schema(q["db_id"], schemas_dir)
            schemas.append(schema)
            schema_text = serialize_schema(schema, filter_question=q["question"] if filter_schema else None)
            prompts.append(format_prompt(q["question"], q["db_id"], schema_text))

        raw_outputs = run_batch(model, tokenizer, prompts)

        for q, schema, raw in zip(batch, schemas, raw_outputs):
            json_str = extract_json(raw)
            try:
                links = json.loads(json_str) if json_str else {}
                if not isinstance(links, dict):
                    links = {}
            except json.JSONDecodeError:
                links = {}

            links = validate_links(links, schema)
            preds.append({"question_id": q["question_id"], "schema_links": links})
        print(f"  {min(i + BATCH_SIZE, len(questions))}/{len(questions)} done")
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--schemas_dir", default=DEFAULT_SCHEMAS_DIR)
    ap.add_argument("--adapter_path", default=ADAPTER_PATH)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--filter_schema", action="store_true",
                    help="Only serialize tables matching question tokens (use on low VRAM)")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Loading model {BASE_MODEL} + adapter {args.adapter_path}...")
    model, tokenizer = load_model(args.adapter_path)

    print(f"Running inference on {len(questions)} questions...")
    preds = predict_all(questions, args.schemas_dir, model, tokenizer, batch_size=args.batch_size, filter_schema=args.filter_schema)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(preds)} predictions to {args.output}")


if __name__ == "__main__":
    main()
