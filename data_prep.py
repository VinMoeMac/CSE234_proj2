import json
import os
import re


# Human-readable descriptions for cryptic table names
# These help the model connect NL concepts to opaque abbreviations
TABLE_DESCRIPTIONS = {
    # NTSB crash investigation tables
    "GV": "General Vehicle info: make, model, year, body type, lighting condition, road surface, weather, speed limit",
    "CRASH": "Crash-level data: crash year, month, day of week, time, number of vehicles",
    "OCC": "Occupant data: age, sex, seat location, belt use, injury severity, height, weight",
    "AVOID": "Pre-crash avoidance maneuvers and collision avoidance equipment: availability, activation",
    "ADAPT": "Adaptive equipment for disabled drivers installed in vehicle",
    "CDC": "Crush Data Coding: crush depth, principal direction of force, delta-v speed",
    "EDREVENT": "Event Data Recorder events: event number, description, ignition cycle",
    "ICS": "Injury Coding Source: body region index, source of energy, confidence",
    "VPICDECODE": "VIN decode: vehicle type, manufacturer, make, model, model year",
    "EDRPRECRASH": "EDR pre-crash data: parameter codes, times, values before crash",
    "TIREPLAC": "Tire placard: recommended front/rear tire sizes and pressures",
    # SAP Business One tables
    "OHEM": "Employee master record: name, salary, department, contact info",
    "OHTY": "Employee type definitions: type ID, name, description",
    "OHTM": "Team master: team ID, name, description",
    "HTM1": "Team members: team ID, employee ID, role",
    "HEM6": "Employee roles: employee ID, role ID",
    "RDOC": "Report documents: document name, author, email settings, extension error action",
    "OUQR": "User queries: query name, category, query string, last date",
    "OQAG": "Query authorization groups: group ID, code, name",
    "OCTR": "Service contracts: contract ID, customer, status, template",
    "OSCL": "Service calls: call ID, subject, customer, status",
    "OQUE": "Service queues: queue ID, description, manager",
    "OJDT": "Journal entries: transaction ID, date, memo",
    "OACT": "Chart of accounts: account code, name, balance",
    "ORCR": "Recurring transactions: code, description, frequency",
    "OPR1": "Sales opportunity details: opportunity ID, open/close dates, closing percentage, amounts",
    "OOST": "Sales opportunity stages: stage number, description, closing percentage",
    "ORCT": "Incoming payments: document number, date, cash amount",
    "OVPM": "Outgoing payments: document number, vendor code, date, total",
    "OCHO": "Checks: check key, account number, check date, amount",
    "OCRD": "Business partners: card code, name, phone, email, balance, credit limit",
    "OSLP": "Sales employees: code, name",
    "OWHS": "Warehouses: code, name, location",
    "OITM": "Items/products: item code, name, stock quantity, price",
    "OWOR": "Production orders: document number, item code, planned quantity, status",
}

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


def split_identifier(name: str) -> set[str]:
    """Split camelCase/PascalCase/snake_case identifiers into tokens."""
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", spaced)
    return set(re.sub(r"[^a-z0-9]", " ", spaced.lower()).split())


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

    # Sort tables by column relevance to question (helps cryptic table names like SAP)
    # Tables whose columns match the question come first
    table_order = list(table_cols.keys())
    if filter_question:
        q_tokens = set(re.sub(r"[^a-z0-9]", " ", filter_question.lower()).split())
        def table_score(t: str) -> int:
            col_tokens = set()
            for cn in table_col_names[t]:
                col_tokens |= split_identifier(cn)
            t_tokens = split_identifier(t)
            return len((col_tokens | t_tokens) & q_tokens)
        table_order = sorted(table_order, key=table_score, reverse=True)

    lines = []
    for table in table_order:
        cols = table_cols[table]
        if table not in allowed_tables:
            continue
        # Table description: use hardcoded TABLE_DESCRIPTIONS first, then derived keywords
        hint = ""
        if table in TABLE_DESCRIPTIONS:
            hint = f"  [{TABLE_DESCRIPTIONS[table]}]"
        elif table_descriptions and table in table_descriptions:
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
    # Sort tables and columns for stable decoding
    sorted_links = {t: sorted(cols) for t, cols in sorted(schema_links.items())}
    answer = json.dumps(sorted_links, ensure_ascii=False)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_prompt(question, db_id, schema_text)},
            {"role": "assistant", "content": answer},
        ]
    }


COLUMN_SYSTEM_PROMPT = (
    "You are a column linking assistant. Given a natural language question and a single "
    "database table with its columns, output a JSON list of column names that the question "
    "references. If no specific columns are referenced (e.g. COUNT(*) or SELECT *), output "
    "an empty list []. Output only a valid JSON array — no explanation, no extra text."
)


def build_two_stage_dataset(data_path: str, schemas_dir: str) -> list[dict]:
    """Build training data for Stage 2: single-table column prediction.
    Each training example becomes one entry per table referenced in the question."""
    with open(data_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    dataset = []
    for ex in examples:
        schema = load_schema(ex["db_id"], schemas_dir)
        table_col_map = {}
        for tbl_idx, col_name in schema["column_names_original"]:
            if tbl_idx == -1:
                continue
            t = schema["table_names_original"][tbl_idx]
            table_col_map.setdefault(t, []).append(col_name)

        for table, gold_cols in ex["schema_links"].items():
            all_cols = table_col_map.get(table, [])
            if not all_cols:
                continue
            col_list = "\n".join(f"  - {c}" for c in all_cols)
            prompt = (
                f"Question: {ex['question']}\n\n"
                f"Table: {table}\n"
                f"Columns:\n{col_list}\n\n"
                f"Which columns from this table does the question reference? "
                f"Output a JSON array of column names, or [] if none."
            )
            answer = json.dumps(gold_cols, ensure_ascii=False)
            dataset.append({
                "messages": [
                    {"role": "system", "content": COLUMN_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ]
            })
    return dataset


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
