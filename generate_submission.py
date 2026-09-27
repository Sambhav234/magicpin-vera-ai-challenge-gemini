"""
magicpin AI Challenge — Submission Generator
============================================
Generates the canonical 30-line `submission.jsonl` from `expanded/test_pairs.json`.
Each line corresponds to one canonical test pair evaluated by the AI Judge.
"""

import json
import os
from pathlib import Path
from bot import compose


def generate_submission(base_dir: str = "."):
    base_path = Path(base_dir)
    expanded_path = base_path / "expanded"

    test_pairs_file = expanded_path / "test_pairs.json"
    if not test_pairs_file.exists():
        raise FileNotFoundError(f"Could not find {test_pairs_file}. Please run dataset expansion first.")

    with open(test_pairs_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = data.get("pairs", data if isinstance(data, list) else [])
    print(f"Loaded {len(pairs)} test pairs from {test_pairs_file}")

    # Load categories cache
    categories = {}
    cat_dir = expanded_path / "categories"
    if cat_dir.exists():
        for cf in cat_dir.glob("*.json"):
            with open(cf, "r", encoding="utf-8") as f:
                cdata = json.load(f)
                categories[cdata.get("slug", cf.stem)] = cdata

    out_file = base_path / "submission.jsonl"
    lines = []

    for item in pairs:
        test_id = item.get("test_id")
        tid = item.get("trigger_id")
        mid = item.get("merchant_id")
        cid = item.get("customer_id")

        # Load trigger
        trigger_path = expanded_path / "triggers" / f"{tid}.json"
        if not trigger_path.exists():
            print(f"[WARN] Trigger file not found: {trigger_path}")
            trigger = {"id": tid, "kind": "generic", "payload": {}}
        else:
            with open(trigger_path, "r", encoding="utf-8") as f:
                trigger = json.load(f)

        # Load merchant
        merchant_path = expanded_path / "merchants" / f"{mid}.json"
        if not merchant_path.exists():
            print(f"[WARN] Merchant file not found: {merchant_path}")
            merchant = {"merchant_id": mid, "identity": {"name": "Merchant"}}
        else:
            with open(merchant_path, "r", encoding="utf-8") as f:
                merchant = json.load(f)

        # Load customer if applicable
        customer = None
        if cid:
            customer_path = expanded_path / "customers" / f"{cid}.json"
            if customer_path.exists():
                with open(customer_path, "r", encoding="utf-8") as f:
                    customer = json.load(f)

        # Category
        cat_slug = merchant.get("category_slug") or trigger.get("payload", {}).get("category") or "dentists"
        category = categories.get(cat_slug, {})

        # Compose message
        result = compose(category, merchant, trigger, customer)

        submission_line = {
            "test_id": test_id,
            "trigger_id": tid,
            "merchant_id": mid,
            "customer_id": cid,
            "body": result.get("body", ""),
            "cta": result.get("cta", "binary"),
            "send_as": result.get("send_as", "vera"),
            "suppression_key": result.get("suppression_key", f"supp_{tid}"),
            "rationale": result.get("rationale", "")
        }
        lines.append(submission_line)

    with open(out_file, "w", encoding="utf-8") as f:
        for entry in lines:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Successfully generated {len(lines)} canonical lines in {out_file}")


if __name__ == "__main__":
    generate_submission()
