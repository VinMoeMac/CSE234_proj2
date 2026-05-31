import json
import os
import re


DB_ID_TO_FILENAME = {
    "SBODemoUS-Business Partners": "SBODemoUS-Business_Partners",
    "SBODemoUS-Human Resources": "SBODemoUS-Human_Resources",
    "SBODemoUS-Inventory and Production": "SBODemoUS-Inventory_and_Production",
    "SBODemoUS-Sales Opportunities": "SBODemoUS-Sales_Opportunities",
}

SYSTEM_PROMPT = (
    "You are a schema linking assistant. Given a natural language question and a "
    "database schema, output a JSON object mapping each referenced table name to a "
    "list of referenced column names. Use an empty list for tables referenced without "
    "specific columns (e.g. COUNT(*)). Output only valid JSON — no explanation."
)


def load_schema(db_id: str, schemas_dir: str) -> dict:
    filename = DB_ID_TO_FILENAME.get(db_id, db_id)
    path = os.path.join(schemas_dir, f"{filename}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def serialize_schema(schema: dict) -> str:
    tables = schema["table_names_original"]
    columns = schema["column_names_original"]  # [table_idx, col_name], index 0 is [-1, '*']
    types = schema["column_types"]
    pks = set(schema["primary_keys"])
    fk_cols = {fk[0] for fk in schema["foreign_keys"]}

    # group columns by table (skip index 0 which is the synthetic wildcard)
    table_cols = {t: [] for t in tables}
    for i, (tbl_idx, col_name) in enumerate(columns):
        if tbl_idx == -1:
            continue
        annotation = []
        if i in pks:
            annotation.append("PK")
        if i in fk_cols:
            annotation.append("FK")
        col_type = types[i - 1]  # types has no entry for the synthetic [-1,'*'] at index 0
        suffix = f" [{col_type}{',' + ','.join(annotation) if annotation else ''}]"
        table_cols[tables[tbl_idx]].append(f"{col_name}{suffix}")

    lines = []
    for table, cols in table_cols.items():
        lines.append(f"Table: {table}")
        for col in cols:
            lines.append(f"  - {col}")
    return "\n".join(lines)


def format_prompt(question: str, db_id: str, schema_text: str) -> str:
    return (
        f"Database: {db_id}\n"
        f"Schema:\n{schema_text}\n\n"
        f"Question: {question}"
    )


def make_chat_example(question: str, db_id: str, schema_text: str, schema_links: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_prompt(question, db_id, schema_text)},
            {"role": "assistant", "content": json.dumps(schema_links, ensure_ascii=False)},
        ]
    }


def build_dataset(data_path: str, schemas_dir: str) -> list[dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    dataset = []
    for ex in examples:
        schema = load_schema(ex["db_id"], schemas_dir)
        schema_text = serialize_schema(schema)
        chat = make_chat_example(
            question=ex["question"],
            db_id=ex["db_id"],
            schema_text=schema_text,
            schema_links=ex["schema_links"],
        )
        dataset.append(chat)
    return dataset


def save_jsonl(dataset: list[dict], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in dataset:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="Project2/train.json")
    parser.add_argument("--validation", default="Project2/validation.json")
    parser.add_argument("--schemas_dir", default="Project2/schemas")
    parser.add_argument("--out_dir", default="data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Building training set...")
    train_data = build_dataset(args.train, args.schemas_dir)
    save_jsonl(train_data, os.path.join(args.out_dir, "train.jsonl"))
    print(f"  {len(train_data)} examples -> {args.out_dir}/train.jsonl")

    print("Building validation set...")
    val_data = build_dataset(args.validation, args.schemas_dir)
    save_jsonl(val_data, os.path.join(args.out_dir, "validation.jsonl"))
    print(f"  {len(val_data)} examples -> {args.out_dir}/validation.jsonl")
