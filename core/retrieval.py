import json
import logging
import os
import sqlite3
import time
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np

from core.normalization import normalize_fts_query, lemmatize_text, build_query_variants
from core.normalization import normalize_query, tokens as norm_tokens

logger = logging.getLogger("search.retrieval")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.environ.get("DB_PATH", os.path.join(ROOT_DIR, "data", "search.db"))
FAISS_INDEX_PATH = os.environ.get(
    "FAISS_INDEX_PATH",
    os.path.join(ROOT_DIR, "data", "ste_faiss.index"),
)
FAISS_IDS_PATH = os.environ.get(
    "FAISS_IDS_PATH",
    os.path.join(ROOT_DIR, "data", "ste_faiss_ids.json"),
)

_FAISS = {"index": None, "ids": None}
_LEMMA_SCHEMA_READY = False

STOPWORDS = {
    "и","в","на","с","для","по","из","от","до","за","под","над",
    "или","но","а","же","бы","не","ни","то","это","как","так",
    "что","все","при","об","к","о"
}


def ensure_db_ready() -> None:
    """Ensure SQLite DB is present and FTS table exists.

    Args:
        None.

    Returns:
        None.
    """
    if not os.path.exists(DB_PATH):
        from core.build_db import build_db
        build_db()
        return
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("SELECT 1 FROM ste_fts LIMIT 1")
        conn.execute("SELECT 1 FROM company_sectors LIMIT 1")
        conn.close()
    except sqlite3.Error:
        from core.build_db import build_db
        build_db()


def parse_ste_attributes(attr_text: str) -> Dict:
    """Parse STE attributes from serialized form.

    Args:
        attr_text: Serialized attributes string.

    Returns:
        Parsed attributes dict.
    """
    if not attr_text:
        return {}
    parts = [p for p in attr_text.split(";") if ":" in p]
    attrs: Dict[str, str] = {}
    for p in parts:
        key, val = p.split(":", 1)
        key = key.strip()
        val = val.strip()
        if key and val:
            attrs[key] = val
    return attrs


def ensure_lemma_schema(conn: sqlite3.Connection) -> None:
    """Ensure lemma columns and FTS schema exist.

    Args:
        conn: SQLite connection.

    Returns:
        None.
    """
    global _LEMMA_SCHEMA_READY
    if _LEMMA_SCHEMA_READY:
        return
    conn.execute("PRAGMA busy_timeout=5000;")
    cols = [row[1] for row in conn.execute("PRAGMA table_info(ste)").fetchall()]
    needs_name = "name_lemma" not in cols
    needs_cat = "category_lemma" not in cols
    if needs_name:
        conn.execute("ALTER TABLE ste ADD COLUMN name_lemma TEXT")
    if needs_cat:
        conn.execute("ALTER TABLE ste ADD COLUMN category_lemma TEXT")
    if needs_name or needs_cat:
        conn.commit()
        _backfill_lemmas(conn)
    try:
        _ensure_fts_schema(conn)
    except sqlite3.OperationalError as exc:
        logger.warning("fts_schema_locked", extra={"error": str(exc)})
        return
    _ensure_company_sectors(conn)
    _LEMMA_SCHEMA_READY = True


def _backfill_lemmas(conn: sqlite3.Connection, batch: int = 2000) -> None:
    """Backfill lemma columns for existing rows.

    Args:
        conn: SQLite connection.
        batch: Batch size for updates.

    Returns:
        None.
    """
    rows = conn.execute(
        "SELECT id, name, category FROM ste "
        "WHERE name_lemma IS NULL OR name_lemma = '' OR category_lemma IS NULL OR category_lemma = ''"
    ).fetchall()
    updates = []
    for ste_id, name, category in rows:
        updates.append(
            (
                lemmatize_text(name or ""),
                lemmatize_text(category or ""),
                str(ste_id),
            )
        )
        if len(updates) >= batch:
            conn.executemany(
                "UPDATE ste SET name_lemma = ?, category_lemma = ? WHERE id = ?",
                updates,
            )
            updates = []
    if updates:
        conn.executemany(
            "UPDATE ste SET name_lemma = ?, category_lemma = ? WHERE id = ?",
            updates,
        )
    conn.commit()


def _ensure_fts_schema(conn: sqlite3.Connection) -> None:
    """Ensure FTS table includes lemma columns.

    Args:
        conn: SQLite connection.

    Returns:
        None.
    """
    cols = [row[1] for row in conn.execute("PRAGMA table_info(ste_fts)").fetchall()]
    required = {"id", "name", "name_lemma", "category", "category_lemma", "attributes"}
    if not required.issubset(set(cols)):
        try:
            conn.execute("DROP TABLE IF EXISTS ste_fts")
            conn.execute(
                "CREATE VIRTUAL TABLE ste_fts USING fts5("
                "id, name, name_lemma, category, category_lemma, attributes, "
                "content='ste', content_rowid='rowid')"
            )
            conn.execute("INSERT INTO ste_fts(ste_fts) VALUES ('rebuild')")
            conn.commit()
        except sqlite3.OperationalError as exc:
            raise


def _ensure_company_sectors(conn: sqlite3.Connection) -> None:
    """Ensure company_sectors table exists.

    Args:
        conn: SQLite connection.

    Returns:
        None.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS company_sectors ("
        "company_name TEXT PRIMARY KEY, "
        "sectors TEXT)"
    )
    conn.commit()


def search_fts(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 200,
    offset: int = 0,
) -> List[Dict]:
    """Search FTS with lemma-normalized query and return items with bm25 score.

    Args:
        conn: SQLite connection.
        query: Search query.
        limit: Max results.
        offset: Offset for pagination.

    Returns:
        List of items with bm25 score.
    """
    q_lemma = lemmatize_text(query)
    q_fts = normalize_fts_query(q_lemma) or normalize_fts_query(query)
    if not q_fts:
        return []
    rows = conn.execute(
        "SELECT s.id, s.name, s.category, s.attributes, bm25(ste_fts) as bm25 "
        "FROM ste_fts f JOIN ste s ON s.rowid = f.rowid "
        "WHERE ste_fts MATCH ? ORDER BY bm25 LIMIT ? OFFSET ?",
        (q_fts, limit, offset),
    ).fetchall()
    results = []
    for ste_id, name, category, attributes, bm25 in rows:
        results.append(
            {
                "id": str(ste_id),
                "name": name or "",
                "category": category or "",
                "attributes": parse_ste_attributes(attributes or ""),
                "bm25": float(-bm25),  # higher is better
            }
        )
    return results


def search_like(conn: sqlite3.Connection, queries: List[str], limit: int = 200) -> List[Dict]:
    """Fallback LIKE search over name/category.

    Args:
        conn: SQLite connection.
        queries: Search queries.
        limit: Max results.

    Returns:
        List of items.
    """
    def _run(tokens: List[str], use_or: bool) -> List[Dict]:
        if not tokens:
            return []
        op = " OR " if use_or else " AND "
        where = op.join(["(name LIKE ? OR category LIKE ?)"] * len(tokens))
        params: List[str | int] = []
        for t in tokens:
            like = f"%{t}%"
            params.extend([like, like])
        params.append(limit)
        rows = conn.execute(
            f"SELECT id, name, category, attributes FROM ste WHERE {where} LIMIT ?",
            params,
        ).fetchall()
        res = []
        for ste_id, name, category, attributes in rows:
            res.append(
                {
                    "id": str(ste_id),
                    "name": name or "",
                    "category": category or "",
                    "attributes": parse_ste_attributes(attributes or ""),
                    "bm25": 0.0,
                }
            )
        return res

    seen = set()
    results = []
    for q in queries:
        qn = normalize_query(q)
        tokens = [t for t in qn.split() if t]
        if not tokens:
            continue
        for item in _run(tokens, use_or=False):
            if item["id"] in seen:
                continue
            seen.add(item["id"])
            results.append(item)
            if len(results) >= limit:
                return results
        if len(results) < max(5, limit // 10):
            for item in _run(tokens, use_or=True):
                if item["id"] in seen:
                    continue
                seen.add(item["id"])
                results.append(item)
                if len(results) >= limit:
                    return results
    return results


def search_fts_multi(conn: sqlite3.Connection, query: str, limit: int = 200) -> List[Dict]:
    """Search FTS over query variants with deduplication.

    Args:
        conn: SQLite connection.
        query: Search query.
        limit: Max results.

    Returns:
        List of items.
    """
    variants = build_query_variants(query)
    results = []
    seen = set()
    for q in variants:
        for item in search_fts(conn, q, limit=limit, offset=0):
            if item["id"] in seen:
                continue
            seen.add(item["id"])
            results.append(item)
            if len(results) >= limit:
                return results
    return results


def load_faiss():
    """Load FAISS index and id mapping lazily.

    Args:
        None.

    Returns:
        Tuple of (index, ids) or (None, None).
    """
    if _FAISS["index"] is not None:
        return _FAISS["index"], _FAISS["ids"]
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(FAISS_IDS_PATH):
        logger.warning("faiss_index_missing", extra={"path": FAISS_INDEX_PATH})
        return None, None
    try:
        import faiss  # type: ignore
    except ImportError:
        logger.warning("faiss_not_installed")
        return None, None
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_IDS_PATH, "r", encoding="utf-8") as f:
        ids = json.load(f)
    _FAISS["index"] = index
    _FAISS["ids"] = ids
    return index, ids


def search_faiss(query_vec: np.ndarray, top_k: int = 50) -> List[Tuple[str, float]]:
    """Search FAISS index with normalized vector, return (id, score).

    Args:
        query_vec: Query embedding.
        top_k: Top K results.

    Returns:
        List of (id, score).
    """
    index, ids = load_faiss()
    if index is None or ids is None:
        return []
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    try:
        scores, idx = index.search(query_vec.astype("float32"), top_k)
    except Exception as exc:
        logger.warning("faiss_search_failed", extra={"error": str(exc)})
        return []
    results = []
    for i, score in zip(idx[0], scores[0]):
        if i < 0 or i >= len(ids):
            continue
        results.append((str(ids[i]), float(score)))
    return results


def compute_contract_stats_sql(conn: sqlite3.Connection, top_n: int = 5000, top_companies: int = 500) -> Dict:
    """Compute contract statistics from SQLite.

    Args:
        conn: SQLite connection.
        top_n: Number of top STE ids.
        top_companies: Number of top companies.

    Returns:
        Dict with stats.
    """
    total_row = conn.execute("SELECT COUNT(1) FROM contracts").fetchone()
    total = int(total_row[0]) if total_row and total_row[0] is not None else 0

    rows = conn.execute(
        "SELECT ste_id, COUNT(1) as cnt FROM contracts GROUP BY ste_id ORDER BY cnt DESC LIMIT ?",
        (top_n,),
    ).fetchall()
    top_ste_ids = [str(sid) for sid, _ in rows]
    top_ste_counts = {str(sid): int(cnt) for sid, cnt in rows}

    customer_rows = conn.execute(
        "SELECT customer_name, COUNT(1) as cnt FROM contracts WHERE customer_name != '' "
        "GROUP BY customer_name ORDER BY cnt DESC LIMIT ?",
        (top_companies,),
    ).fetchall()
    supplier_rows = conn.execute(
        "SELECT supplier_name, COUNT(1) as cnt FROM contracts WHERE supplier_name != '' "
        "GROUP BY supplier_name ORDER BY cnt DESC LIMIT ?",
        (top_companies,),
    ).fetchall()

    return {
        "total_contract_rows": total,
        "top_ste_ids": top_ste_ids,
        "top_ste_counts": top_ste_counts,
        "top_customers": [n for n, _ in customer_rows],
        "top_suppliers": [n for n, _ in supplier_rows],
    }


def load_contract_names_sql(conn: sqlite3.Connection, ste_ids: Iterable[str]) -> Dict[str, str]:
    """Load item names for STE ids from contracts.

    Args:
        conn: SQLite connection.
        ste_ids: Iterable of STE ids.

    Returns:
        Mapping ste_id -> item_name.
    """
    names: Dict[str, str] = {}
    ids = list(ste_ids)
    if not ids:
        return names
    for batch in _chunked(ids, size=900):
        placeholders = ",".join(["?"] * len(batch))
        rows = conn.execute(
            f"SELECT ste_id, MIN(item_name) FROM contracts "
            f"WHERE ste_id IN ({placeholders}) AND item_name != '' GROUP BY ste_id",
            batch,
        ).fetchall()
        for ste_id, item_name in rows:
            if item_name:
                names[str(ste_id)] = item_name
    return names


def build_company_profiles_sql(
    conn: sqlite3.Connection,
    companies: Iterable[str],
    kind: str = "customer",
    limit_ids: int = 200,
) -> Dict[str, List[str]]:
    """Build per-company top STE ids.

    Args:
        conn: SQLite connection.
        companies: Company names.
        kind: customer|supplier.
        limit_ids: Max ids per company.

    Returns:
        Mapping company -> list of ste_ids.
    """
    company_to_counts: Dict[str, Dict[str, int]] = {c: {} for c in companies}
    if not company_to_counts:
        return {}
    col = "customer_name" if kind == "customer" else "supplier_name"
    profiles: Dict[str, List[str]] = {}
    for name in companies:
        rows = conn.execute(
            f"SELECT ste_id, COUNT(1) as cnt FROM contracts WHERE {col} = ? "
            f"GROUP BY ste_id ORDER BY cnt DESC LIMIT ?",
            (name, limit_ids),
        ).fetchall()
        profiles[name] = [str(ste_id) for ste_id, _ in rows]
    return profiles


def build_ste_index_sql(conn: sqlite3.Connection, needed_ids: Iterable[str]) -> Dict[str, Dict]:
    """Load STE rows for given ids.

    Args:
        conn: SQLite connection.
        needed_ids: Iterable of STE ids.

    Returns:
        Mapping ste_id -> item dict.
    """
    index: Dict[str, Dict] = {}
    ids = list(needed_ids)
    if not ids:
        return index
    for batch in _chunked(ids, size=900):
        placeholders = ",".join(["?"] * len(batch))
        rows = conn.execute(
            f"SELECT id, name, category, attributes FROM ste WHERE id IN ({placeholders})",
            batch,
        ).fetchall()
        for ste_id, name, category, attributes in rows:
            index[str(ste_id)] = {
                "id": str(ste_id),
                "name": name or "",
                "category": category or "",
                "attributes": attributes or "",
            }
    return index


def top_categories_sql(conn: sqlite3.Connection, limit: int = 2000) -> List[str]:
    """Load top categories by frequency.

    Args:
        conn: SQLite connection.
        limit: Max number of categories.

    Returns:
        List of categories.
    """
    rows = conn.execute(
        "SELECT category, COUNT(1) as cnt FROM ste WHERE category != '' "
        "GROUP BY category ORDER BY cnt DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [c for c, _ in rows]


def fetch_item_sql(conn: sqlite3.Connection, item_id: str) -> Optional[Dict]:
    """Fetch single item by id.

    Args:
        conn: SQLite connection.
        item_id: STE id.

    Returns:
        Item dict or None.
    """
    row = conn.execute(
        "SELECT id, name, category, attributes FROM ste WHERE id = ? LIMIT 1",
        (str(item_id),),
    ).fetchone()
    if not row:
        return None
    ste_id, name, category, attributes = row
    return {
        "id": str(ste_id),
        "name": name or "",
        "category": category or "",
        "attributes": parse_ste_attributes(attributes or ""),
    }


def get_company_sectors(conn: sqlite3.Connection, company_name: str) -> List[str]:
    """Get stored sectors for a company.

    Args:
        conn: SQLite connection.
        company_name: Company name.

    Returns:
        List of sector labels.
    """
    if not company_name:
        return []
    row = conn.execute(
        "SELECT sectors FROM company_sectors WHERE company_name = ? LIMIT 1",
        (company_name,),
    ).fetchone()
    if not row or not row[0]:
        return []
    try:
        return json.loads(row[0])
    except Exception:
        return []


def set_company_sectors(conn: sqlite3.Connection, company_name: str, sectors: List[str]) -> None:
    """Store sectors for a company.

    Args:
        conn: SQLite connection.
        company_name: Company name.
        sectors: Sector list.

    Returns:
        None.
    """
    if not company_name:
        return
    payload = json.dumps(sectors, ensure_ascii=False)
    conn.execute(
        "INSERT INTO company_sectors(company_name, sectors) "
        "VALUES(?, ?) ON CONFLICT(company_name) DO UPDATE SET sectors=excluded.sectors",
        (company_name, payload),
    )
    conn.commit()


def company_category_stats(conn: sqlite3.Connection, company_name: str, kind: str = "customer", limit: int = 50) -> List[str]:
    """Return top category names for a company from contracts.

    Args:
        conn: SQLite connection.
        company_name: Company name.
        kind: customer|supplier.
        limit: Max categories.

    Returns:
        List of category names.
    """
    if not company_name:
        return []
    col = "customer_name" if kind == "customer" else "supplier_name"
    rows = conn.execute(
        f"SELECT s.category, COUNT(1) as cnt "
        f"FROM contracts c JOIN ste s ON s.id = c.ste_id "
        f"WHERE c.{col} = ? AND s.category != '' "
        f"GROUP BY s.category ORDER BY cnt DESC LIMIT ?",
        (company_name, limit),
    ).fetchall()
    return [r[0] for r in rows if r and r[0]]


def _chunked(values: Iterable[str], size: int = 900) -> Iterable[List[str]]:
    """Yield chunks from an iterable.

    Args:
        values: Iterable of strings.
        size: Chunk size.

    Returns:
        Iterable of lists.
    """
    buf: List[str] = []
    for v in values:
        buf.append(v)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def expand_query_from_session(query: str, session_feedback: Dict, db_conn: sqlite3.Connection) -> str:
    """Expand query using session item tokens."""
    t0 = time.perf_counter()
    if not session_feedback:
        return query
    q_tokens = norm_tokens(lemmatize_text(query))
    short_query = len(q_tokens) <= 1
    visited = set(session_feedback.get("visited_ids", []) or [])
    liked = set(session_feedback.get("liked_ids", []) or [])
    dwell = set(session_feedback.get("long_dwell_ids", []) or [])
    session_ids = [str(x) for x in (visited | liked | dwell) if str(x)]
    if not session_ids:
        return query
    names: List[str] = []
    for batch in _chunked(session_ids, size=900):
        placeholders = ",".join(["?"] * len(batch))
        rows = db_conn.execute(
            f"SELECT name FROM ste WHERE id IN ({placeholders})",
            batch,
        ).fetchall()
        for (name,) in rows:
            if name:
                names.append(name)
    if not names:
        return query
    token_counts: Dict[str, int] = {}
    for name in names:
        lemmas = lemmatize_text(name)
        for tok in norm_tokens(lemmas):
            if tok in STOPWORDS:
                continue
            if not tok.isalpha():
                continue
            if any(ch.isdigit() for ch in tok):
                continue
            token_counts[tok] = token_counts.get(tok, 0) + 1
    if not token_counts:
        return query
    query_lemmas = set(q_tokens)
    # If query has no overlap with session and is not short, skip expansion
    if not short_query and not (set(token_counts.keys()) & query_lemmas):
        return query
    top_tokens = [
        t for t, _ in sorted(token_counts.items(), key=lambda x: (-x[1], x[0]))
        if t not in query_lemmas
    ][:3]
    if not top_tokens:
        return query
    expanded = f"{query} {' '.join(top_tokens)}".strip()
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "session_query_expanded",
            extra={"original": query, "expanded": expanded},
        )
    _ = time.perf_counter() - t0
    return expanded
