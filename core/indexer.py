import argparse
import json
import os
import pickle
import sqlite3
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp

from core.retrieval import ensure_db_ready, ensure_lemma_schema, DB_PATH, FAISS_INDEX_PATH, FAISS_IDS_PATH

E5_MODEL = "intfloat/multilingual-e5-small"
_PASSAGE_PREFIX = "passage: "


def _load_ste_rows(conn: sqlite3.Connection) -> List[Tuple[str, str, str]]:
    """Load STE rows for indexing.

    Args:
        conn: SQLite connection.

    Returns:
        List of (id, name, category).
    """
    rows = conn.execute("SELECT id, name, category FROM ste").fetchall()
    return [(str(ste_id), name or "", category or "") for ste_id, name, category in rows]


def build_faiss() -> None:
    """Build FAISS index over STE names+categories.

    Args:
        None.

    Returns:
        None.
    """
    ensure_db_ready()
    conn = sqlite3.connect(DB_PATH)
    ensure_lemma_schema(conn)
    rows = _load_ste_rows(conn)
    conn.close()
    max_items_raw = os.environ.get("FAISS_MAX_ITEMS")
    if max_items_raw:
        try:
            max_items = max(1, int(max_items_raw))
            rows = rows[:max_items]
        except ValueError:
            pass

    try:
        from sentence_transformers import SentenceTransformer
        import faiss  # type: ignore
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}")

    model_path = os.environ.get("ST_MODEL_PATH", E5_MODEL)
    if os.path.isdir(model_path):
        model = SentenceTransformer(model_path, device="cpu")
    else:
        model = SentenceTransformer(E5_MODEL)
    texts = [f"{_PASSAGE_PREFIX}{name} {category}".strip() for _, name, category in rows]
    vectors = model.encode(texts, show_progress_bar=True, batch_size=512, normalize_embeddings=True)
    vectors = np.asarray(vectors, dtype="float32")
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    ids = [ste_id for ste_id, _, _ in rows]
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(FAISS_IDS_PATH, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False)


def build_cf() -> None:
    """Build item-item cosine CF map from contracts.

    Args:
        None.

    Returns:
        None.
    """
    ensure_db_ready()
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT customer_inn, ste_id, COUNT(1) as cnt "
        "FROM contracts WHERE customer_inn != '' AND ste_id != '' "
        "GROUP BY customer_inn, ste_id"
    ).fetchall()
    conn.close()

    customers = {}
    items = {}
    data = []
    row_idx = []
    col_idx = []
    for cust, ste_id, cnt in rows:
        if cust not in customers:
            customers[cust] = len(customers)
        if ste_id not in items:
            items[ste_id] = len(items)
        row_idx.append(items[ste_id])
        col_idx.append(customers[cust])
        data.append(float(cnt))

    if not items:
        raise SystemExit("No contracts data to build CF.")

    item_user = sp.coo_matrix((data, (row_idx, col_idx)), shape=(len(items), len(customers))).tocsr()

    try:
        from implicit.nearest_neighbours import CosineRecommender
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}")

    model = CosineRecommender(K=50)
    model.fit(item_user)

    id_by_index = {idx: ste_id for ste_id, idx in items.items()}
    cf_map: Dict[str, List[Tuple[str, float]]] = {}
    for item_idx in range(item_user.shape[0]):
        similar = model.similar_items(item_idx, N=50)
        ste_id = id_by_index[item_idx]
        cf_map[ste_id] = [(id_by_index[i], float(score)) for i, score in similar if i != item_idx]

    out_path = os.path.join(os.path.dirname(DB_PATH), "item_cf.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(cf_map, f)


def main() -> None:
    """CLI entrypoint.

    Args:
        None.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build_faiss", "build_cf"])
    args = parser.parse_args()
    if args.command == "build_faiss":
        build_faiss()
    elif args.command == "build_cf":
        build_cf()


if __name__ == "__main__":
    main()
