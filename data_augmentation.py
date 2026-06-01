"""
Augment training data by paraphrasing NL questions using Claude API.
Each example gets N paraphrases — same schema_links, different question wording.
Run offline before training; results are committed to data/augmented/.
"""
import argparse
import json
import os
import time

import anthropic


SYSTEM_PROMPT = (
    "You are a paraphrasing assistant. Given a natural language question about a database, "
    "rewrite it in a different way that preserves the exact meaning and intent. "
    "Output only the rewritten question — no explanation, no prefix, no quotes."
)


def paraphrase_question(client: anthropic.Anthropic, question: str, n: int = 3) -> list[str]:
    """Generate n paraphrases of a question using Claude."""
    results = []
    for i in range(n):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=256,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": f"Paraphrase this database question (variation {i+1} of {n}, make each variation distinct):\n\n{question}"
                }],
            )
            paraphrase = response.content[0].text.strip()
            if paraphrase and paraphrase != question:
                results.append(paraphrase)
        except Exception as e:
            print(f"  Warning: paraphrase {i+1} failed: {e}")
            time.sleep(1)
    return results


def augment_dataset(
    data_path: str,
    out_path: str,
    n_paraphrases: int = 3,
    max_examples: int | None = None,
) -> None:
    with open(data_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    if max_examples:
        examples = examples[:max_examples]

    client = anthropic.Anthropic()

    augmented = list(examples)  # keep originals
    total = len(examples)

    for i, ex in enumerate(examples):
        print(f"  [{i+1}/{total}] {ex['question'][:60]}...")
        paraphrases = paraphrase_question(client, ex["question"], n=n_paraphrases)
        for para in paraphrases:
            augmented.append({
                "question_id": ex["question_id"],
                "db_id": ex["db_id"],
                "question": para,
                "gold_sql": ex.get("gold_sql", ""),
                "schema_links": ex["schema_links"],
            })
        if (i + 1) % 10 == 0:
            print(f"    Progress: {i+1}/{total} done, {len(augmented)} total examples so far")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(augmented, f, indent=2, ensure_ascii=False)

    print(f"\nDone: {len(examples)} original + {len(augmented) - len(examples)} paraphrases = {len(augmented)} total")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="Project2/train.json")
    ap.add_argument("--out", default="data/augmented/train_augmented.json")
    ap.add_argument("--n_paraphrases", type=int, default=3)
    ap.add_argument("--max_examples", type=int, default=None,
                    help="Limit for testing (default: all)")
    args = ap.parse_args()

    print(f"Augmenting {args.train} with {args.n_paraphrases} paraphrases per example...")
    augment_dataset(
        data_path=args.train,
        out_path=args.out,
        n_paraphrases=args.n_paraphrases,
        max_examples=args.max_examples,
    )
