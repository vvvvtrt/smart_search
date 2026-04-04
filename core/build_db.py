#!/usr/bin/env python3
import csv
import os
import sqlite3

from core.normalization import lemmatize_text

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STE_PATH = os.environ.get("DATA_STE_PATH", os.path.join(ROOT_DIR, "data", "СТЕ_20260403.csv"))
CONTRACTS_PATH = os.environ.get("DATA_CONTRACTS_PATH", os.path.join(ROOT_DIR, "data", "Контракты_20260403.csv"))
DB_PATH = os.environ.get("DB_PATH", os.path.join(ROOT_DIR, "data", "search.db"))


def stream_csv(path):
    with open(path, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter=";", quotechar='"')
        for row in reader:
            if row:
                yield row


def build_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    try:
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
    except FileNotFoundError:
        pass
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    conn.execute(
        "CREATE TABLE ste ("
        "id TEXT PRIMARY KEY, "
        "name TEXT, "
        "name_lemma TEXT, "
        "category TEXT, "
        "category_lemma TEXT, "
        "attributes TEXT)"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE ste_fts USING fts5("
        "id, name, name_lemma, category, category_lemma, attributes, "
        "content='ste', content_rowid='rowid')"
    )
    conn.execute(
        "CREATE TABLE contracts (item_name TEXT, contract_id TEXT, ste_id TEXT, created_at TEXT, price TEXT, customer_inn TEXT, customer_name TEXT, customer_region TEXT, supplier_inn TEXT, supplier_name TEXT, supplier_region TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS company_sectors (company_name TEXT PRIMARY KEY, sectors TEXT)"
    )

    ste_rows = []
    for row in stream_csv(STE_PATH):
        if len(row) < 3:
            continue
        ste_id = row[0].strip()
        name = row[1].strip() if len(row) > 1 else ""
        category = row[2].strip() if len(row) > 2 else ""
        attributes = row[3].strip() if len(row) > 3 else ""
        name_lemma = lemmatize_text(name or "")
        category_lemma = lemmatize_text(category or "")
        ste_rows.append((ste_id, name, name_lemma, category, category_lemma, attributes))
        if len(ste_rows) >= 5000:
            conn.executemany(
                "INSERT OR IGNORE INTO ste VALUES (?,?,?,?,?,?)",
                ste_rows,
            )
            ste_rows = []
    if ste_rows:
        conn.executemany(
            "INSERT OR IGNORE INTO ste VALUES (?,?,?,?,?,?)",
            ste_rows,
        )

    conn.execute("INSERT INTO ste_fts(ste_fts) VALUES ('rebuild')")

    contract_rows = []
    for row in stream_csv(CONTRACTS_PATH):
        if len(row) < 11:
            continue
        contract_rows.append(tuple(row[:11]))
        if len(contract_rows) >= 5000:
            conn.executemany("INSERT INTO contracts VALUES (?,?,?,?,?,?,?,?,?,?,?)", contract_rows)
            contract_rows = []
    if contract_rows:
        conn.executemany("INSERT INTO contracts VALUES (?,?,?,?,?,?,?,?,?,?,?)", contract_rows)

    conn.execute("CREATE INDEX idx_contracts_ste ON contracts(ste_id)")
    conn.execute("CREATE INDEX idx_contracts_customer ON contracts(customer_name)")
    conn.execute("CREATE INDEX idx_contracts_supplier ON contracts(supplier_name)")

    conn.commit()
    conn.close()
    print(f"DB created: {DB_PATH}")


def main():
    build_db()


if __name__ == "__main__":
    main()
