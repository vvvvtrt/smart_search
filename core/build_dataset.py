#!/usr/bin/env python3
import argparse
import os
import random
import sqlite3
import csv
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.retrieval import ensure_db_ready, DB_PATH

DB_PATH = os.environ.get("DB_PATH", os.path.join(ROOT_DIR, "data", "search.db"))
OUT_PATH = os.environ.get("DATASET_PATH", os.path.join(ROOT_DIR, "data", "train.csv"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-positives", type=int, default=200000)
    parser.add_argument("--neg-per-pos", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    ensure_db_ready()
    conn = sqlite3.connect(DB_PATH)

    # Build category -> ids
    cat_map = {}
    for row in conn.execute("SELECT id, category FROM ste"):
        ste_id, cat = row
        if cat is None:
            continue
        cat_map.setdefault(cat, []).append(str(ste_id))

    positives = conn.execute(
        "SELECT item_name, ste_id, customer_name FROM contracts WHERE ste_id IS NOT NULL LIMIT ?",
        (args.max_positives,),
    ).fetchall()
    if not positives:
        conn.close()
        raise SystemExit("No positives found. Check that search.db is built and contracts are loaded.")

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "label",
                "query_text",
                "item_id",
                "item_name",
                "item_category",
                "company_name",
            ]
        )

        for item_name, ste_id, company in positives:
            row = conn.execute(
                "SELECT name, category FROM ste WHERE id = ? LIMIT 1",
                (str(ste_id),),
            ).fetchone()
            if not row:
                continue
            name, category = row
            query_text = (item_name or name or category or "")
            writer.writerow([1, query_text, ste_id, name or "", category or "", company or ""])

            # negatives from same category
            pool = cat_map.get(category, [])
            if not pool:
                continue
            for _ in range(args.neg_per_pos):
                neg_id = random.choice(pool)
                if str(neg_id) == str(ste_id):
                    continue
                nrow = conn.execute(
                    "SELECT name, category FROM ste WHERE id = ? LIMIT 1",
                    (str(neg_id),),
                ).fetchone()
                if not nrow:
                    continue
                nname, ncat = nrow
                writer.writerow([0, query_text, neg_id, nname or "", ncat or "", company or ""])

    conn.close()
    print(f"Dataset saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
