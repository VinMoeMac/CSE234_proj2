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
    "You are a schema linking assistant. Given a natural language question and a database schema, "
    "identify which tables and columns the underlying SQL query would reference.\n\n"
    "Rules:\n"
    "- Output a JSON object mapping each referenced table name to a list of referenced column names.\n"
    "- Use the exact table and column names from the schema (case matters for readability).\n"
    "- Include a table with an empty list [] if it is referenced but no specific columns are needed (e.g. COUNT(*), SELECT *).\n"
    "- Include columns used in WHERE, GROUP BY, ORDER BY, JOIN ON, and SELECT clauses.\n"
    "- Do NOT include tables or columns not present in the schema.\n"
    "- Output only valid JSON — no explanation, no markdown, no extra text."
)


def load_schema(db_id: str, schemas_dir: str) -> dict:
    filename = DB_ID_TO_FILENAME.get(db_id, db_id)
    path = os.path.join(schemas_dir, f"{filename}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_tables(schema: dict, question: str) -> list[str]:
    """Return table names that have at least one token matching the question (case-insensitive).
    Always keeps tables with PK/FK relationships to retained tables to preserve join paths."""
    tables = schema["table_names_original"]
    q_tokens = set(re.sub(r"[^a-z0-9]", " ", question.lower()).split())

    fk_pairs = schema["foreign_keys"]  # [[col_idx, col_idx], ...]
    columns = schema["column_names_original"]

    # map col index -> table index
    col_to_table = {i: tbl_idx for i, (tbl_idx, _) in enumerate(columns) if tbl_idx != -1}

    # find tables with a keyword match
    matched = set()
    for t in tables:
        t_tokens = set(re.sub(r"[^a-z0-9]", " ", t.lower()).split())
        if t_tokens & q_tokens:
            matched.add(t)

    # expand: keep tables connected via FK to any matched table
    changed = True
    while changed:
        changed = False
        for col_a, col_b in fk_pairs:
            ta = tables[col_to_table[col_a]] if col_a in col_to_table else None
            tb = tables[col_to_table[col_b]] if col_b in col_to_table else None
            if ta in matched and tb not in matched:
                matched.add(tb)
                changed = True
            elif tb in matched and ta not in matched:
                matched.add(ta)
                changed = True

    # fall back to all tables if nothing matched
    return list(matched) if matched else tables


def build_table_descriptions(train_path: str) -> dict[str, list[str]]:
    """Approach 1: derive table usage descriptions from training questions.
    Returns {db_id: {table: [question snippets that reference it]}}"""
    with open(train_path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    desc: dict[str, dict[str, list[str]]] = {}
    for ex in examples:
        db = ex["db_id"]
        if db not in desc:
            desc[db] = {}
        for table in ex["schema_links"]:
            if table not in desc[db]:
                desc[db][table] = []
            # extract key noun phrases (words >3 chars, not stopwords)
            words = [w for w in re.findall(r"[a-zA-Z]{4,}", ex["question"])
                     if w.lower() not in {"that", "where", "which", "have", "from", "with",
                                          "what", "show", "list", "find", "give", "each",
                                          "this", "there", "were", "been", "when", "then"}]
            if words:
                desc[db][table].extend(words[:4])
    # deduplicate and limit
    result = {}
    for db, tables in desc.items():
        result[db] = {t: list(dict.fromkeys(words))[:8] for t, words in tables.items()}
    return result


def serialize_schema(
    schema: dict,
    filter_question: str | None = None,
    table_descriptions: dict | None = None,
    sort_by_question: bool = False,
    show_fk_links: bool = False,
) -> str:
    """Serialize schema with optional enhancements.

    Args:
        filter_question: if set, only include tables matching question tokens
        table_descriptions: Approach 1 — dict of {table: [keywords]} to prepend as table hints
        sort_by_question: Approach 3 — sort columns by relevance to question
        show_fk_links: Approach 2 — add FK relationship lines between tables
    """
    tables = schema["table_names_original"]
    columns = schema["column_names_original"]  # [table_idx, col_name], index 0 is [-1, '*']
    types = schema["column_types"]
    pks = set(schema["primary_keys"])
    fk_pairs = schema["foreign_keys"]
    fk_cols = {fk[0] for fk in fk_pairs}

    allowed_tables = set(filter_tables(schema, filter_question)) if filter_question else set(tables)

    # group columns by table
    table_cols: dict[str, list[str]] = {t: [] for t in tables}
    table_col_names: dict[str, list[str]] = {t: [] for t in tables}
    for i, (tbl_idx, col_name) in enumerate(columns):
        if tbl_idx == -1:
            continue
        annotation = []
        if i in pks:
            annotation.append("PK")
        if i in fk_cols:
            annotation.append("FK")
        col_type = types[i - 1]
        suffix = f" [{col_type}{',' + ','.join(annotation) if annotation else ''}]"
        table_cols[tables[tbl_idx]].append(f"{col_name}{suffix}")
        table_col_names[tables[tbl_idx]].append(col_name)

    # Approach 3: sort columns by relevance to question
    if sort_by_question and filter_question:
        q_tokens = set(re.sub(r"[^a-z0-9]", " ", filter_question.lower()).split())
        for t in tables:
            pairs = list(zip(table_cols[t], table_col_names[t]))
            pairs.sort(key=lambda x: -len(
                set(re.sub(r"[^a-z0-9]", " ", x[1].lower()).split()) & q_tokens
            ))
            table_cols[t] = [p[0] for p in pairs]

    # Approach 2: build FK link map (table -> list of linked tables, deduplicated)
    fk_links: dict[str, list[str]] = {t: [] for t in tables}
    if show_fk_links:
        col_to_table = {i: tables[tbl_idx]
                        for i, (tbl_idx, _) in enumerate(columns) if tbl_idx != -1}
        seen_links: set[tuple] = set()
        for col_a, col_b in fk_pairs:
            ta = col_to_table.get(col_a)
            tb = col_to_table.get(col_b)
            if ta and tb and ta != tb:
                if (ta, tb) not in seen_links:
                    fk_links[ta].append(tb)
                    seen_links.add((ta, tb))
                if (tb, ta) not in seen_links:
                    fk_links[tb].append(ta)
                    seen_links.add((tb, ta))

    lines = []
    for table, cols in table_cols.items():
        if table not in allowed_tables:
            continue
        # Approach 1: prepend table description hint
        hint = ""
        if table_descriptions and table in table_descriptions:
            keywords = table_descriptions[table]
            if keywords:
                hint = f"  [used for: {', '.join(keywords)}]"
        lines.append(f"Table: {table}{hint}")
        # Approach 2: show FK links
        if show_fk_links and fk_links[table]:
            lines.append(f"  [linked to: {', '.join(fk_links[table])}]")
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


def build_dataset(
    data_path: str,
    schemas_dir: str,
    filter_schema: bool = False,
    table_descriptions: dict | None = None,
    sort_by_question: bool = False,
    show_fk_links: bool = False,
) -> list[dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    dataset = []
    for ex in examples:
        schema = load_schema(ex["db_id"], schemas_dir)
        question = ex["question"] if (filter_schema or sort_by_question) else None
        db_desc = table_descriptions.get(ex["db_id"]) if table_descriptions else None
        schema_text = serialize_schema(
            schema,
            filter_question=question,
            table_descriptions=db_desc,
            sort_by_question=sort_by_question,
            show_fk_links=show_fk_links,
        )
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
    parser.add_argument("--filter_schema", action="store_true",
                        help="Only include tables with tokens matching the question (use on low VRAM)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Building training set...")
    train_data = build_dataset(args.train, args.schemas_dir, filter_schema=args.filter_schema)
    save_jsonl(train_data, os.path.join(args.out_dir, "train.jsonl"))
    print(f"  {len(train_data)} examples -> {args.out_dir}/train.jsonl")

    print("Building validation set...")
    val_data = build_dataset(args.validation, args.schemas_dir, filter_schema=args.filter_schema)
    save_jsonl(val_data, os.path.join(args.out_dir, "validation.jsonl"))
    print(f"  {len(val_data)} examples -> {args.out_dir}/validation.jsonl")
