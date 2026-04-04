#!/usr/bin/env python3
import argparse
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from core.embedding import encode_query
from core.normalization import normalize_query, build_query_variants
from core.personalization import build_behavior_context, cf_candidates
from core.retrieval import (
    DB_PATH,
    ensure_db_ready,
    ensure_lemma_schema,
    compute_contract_stats_sql,
    build_company_profiles_sql,
    load_contract_names_sql,
    build_ste_index_sql,
    top_categories_sql,
    search_fts_multi,
    search_like,
    search_faiss,
    fetch_item_sql,
)
from core.scoring import SearchConfig, behavior_score, personal_reasons, score_candidates
from core.suggestions import (
    autocomplete,
    build_personal_names,
    build_vocab,
    detect_intent,
    expand_queries,
    explanations,
    infer_filters,
    ranking_hints,
    similar_categories,
    suggest_correction,
)
from core.utils import read_json, write_json, file_mtime, load_cache, save_cache

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_PATH = os.environ.get(
    "CACHE_PATH",
    os.path.join(ROOT_DIR, ".search_agent_cache.json"),
)

_MEMO: Dict[str, Any] = {
    "db_mtime": None,
    "top_categories": [],
    "contract_names": {},
    "contract_names_key": None,
}
def run_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run search pipeline.

    Args:
        payload: Request payload.

    Returns:
        Search response dict.
    """
    t0 = time.perf_counter()
    user_query = payload.get("user_query", "")
    user_profile = payload.get("user_profile", {}) or {}
    favorite_categories = user_profile.get("favorite_categories", []) or []
    purchase_history = user_profile.get("purchase_history", []) or []
    recent_actions = user_profile.get("recent_actions", []) or []
    company = payload.get("company") or {}
    company_name = company.get("name") if isinstance(company, dict) else company
    company_type = company.get("type", "customer") if isinstance(company, dict) else "customer"
    mode = payload.get("mode")
    limit = int(payload.get("limit", 5))
    page = int(payload.get("page", 1))
    offset = max(0, (page - 1) * limit)
    session_feedback = payload.get("session_feedback", {}) or {}

    config = SearchConfig(debug=os.environ.get("SEARCH_DEBUG", "0") == "1")

    ensure_db_ready()
    t_db_ready = time.perf_counter()
    conn = sqlite3.connect(DB_PATH)
    ensure_lemma_schema(conn)
    try:
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA cache_size=200000;")
        conn.execute("PRAGMA mmap_size=268435456;")
    except sqlite3.Error:
        pass
    t_db_conn = time.perf_counter()

    cache = load_cache(CACHE_PATH)
    db_mtime = file_mtime(DB_PATH)
    if not cache or cache.get("db_mtime") != db_mtime:
        stats = compute_contract_stats_sql(conn, top_n=5000)
        customer_profiles = build_company_profiles_sql(
            conn,
            stats.get("top_customers", []),
            kind="customer",
            limit_ids=200,
        )
        supplier_profiles = build_company_profiles_sql(
            conn,
            stats.get("top_suppliers", []),
            kind="supplier",
            limit_ids=200,
        )
        cache = {
            "db_mtime": db_mtime,
            "stats": stats,
            "customer_profiles": customer_profiles,
            "supplier_profiles": supplier_profiles,
            "built_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        save_cache(CACHE_PATH, cache)
    else:
        stats = cache["stats"]
        customer_profiles = cache.get("customer_profiles", {})
        supplier_profiles = cache.get("supplier_profiles", {})
    t_stats = time.perf_counter()

    if mode == "companies_only":
        conn.close()
        return {
            "customer_companies": stats.get("top_customers", []),
            "supplier_companies": stats.get("top_suppliers", []),
        }

    needed_ids: Set[str] = set(stats.get("top_ste_ids", []))
    for entry in purchase_history:
        if isinstance(entry, dict):
            ste_id = entry.get("ste_id") or entry.get("id")
            if ste_id:
                needed_ids.add(str(ste_id))
        else:
            if str(entry).isdigit():
                needed_ids.add(str(entry))

    if company_name:
        profile_map = customer_profiles if company_type == "customer" else supplier_profiles
        for ste_id in profile_map.get(company_name, []):
            needed_ids.add(str(ste_id))

    ste_index = build_ste_index_sql(conn, needed_ids)
    if _MEMO.get("db_mtime") != db_mtime:
        _MEMO["db_mtime"] = db_mtime
        _MEMO["top_categories"] = top_categories_sql(conn, limit=2000)
        _MEMO["contract_names"] = {}
        _MEMO["contract_names_key"] = None
    top_categories = _MEMO.get("top_categories", [])
    top_ste_key = tuple(stats.get("top_ste_ids", [])[:2000])
    if _MEMO.get("contract_names_key") != top_ste_key:
        _MEMO["contract_names"] = load_contract_names_sql(conn, top_ste_key)
        _MEMO["contract_names_key"] = top_ste_key
    contract_names = _MEMO.get("contract_names", {})
    t_indices = time.perf_counter()

    popular_names: List[str] = []
    popular_by_category: Dict[str, List[str]] = {}
    for ste_id in stats.get("top_ste_ids", [])[:200]:
        item = ste_index.get(ste_id)
        if item and item["name"]:
            popular_names.append(item["name"])
            if item.get("category"):
                popular_by_category.setdefault(item["category"], []).append(item["name"])
        elif ste_id in contract_names:
            popular_names.append(contract_names[ste_id])

    personal_names = build_personal_names(purchase_history, ste_index, contract_names)
    company_personal_ids: List[str] = []
    if company_name:
        profile_map = customer_profiles if company_type == "customer" else supplier_profiles
        company_personal_ids = profile_map.get(company_name, [])

    semantic_terms: List[str] = []
    for c in similar_categories(user_query, top_categories, limit=20):
        semantic_terms.append(c)
    for a in recent_actions:
        if isinstance(a, str):
            semantic_terms.append(a)

    known_categories = list(dict.fromkeys(favorite_categories + top_categories))

    intent = detect_intent(user_query, known_categories)
    vocab = build_vocab(known_categories, popular_names[:200])
    corrected = suggest_correction(user_query, vocab)
    corrected_query = user_query
    if corrected and corrected != normalize_query(user_query):
        corrected_query = f"Возможно, вы имели в виду: {corrected}"

    category_filter = infer_filters(user_query, known_categories)
    category = category_filter.get("category")

    expanded = expand_queries(user_query, intent, popular_names, personal_names, category)
    preferred_categories = [category] if category else ([] if user_query else favorite_categories)
    sugg = autocomplete(
        user_query,
        popular_names,
        personal_names,
        semantic_terms,
        popular_by_category,
        preferred_categories,
    )

    # Retrieval: FTS + FAISS
    candidates_fts = search_fts_multi(conn, user_query, limit=200)
    if not candidates_fts:
        variants = build_query_variants(user_query)
        if not variants:
            variants = [user_query]
        candidates_fts = search_like(conn, variants, limit=200)
    bm25_scores = {it["id"]: it.get("bm25", 0.0) for it in candidates_fts}

    vector_scores: Dict[str, float] = {}
    vector_items: Dict[str, Dict[str, Any]] = {}
    q_vec = encode_query(user_query)
    if q_vec is not None:
        for ste_id, score in search_faiss(q_vec, top_k=50):
            vector_scores[ste_id] = score
        if vector_scores:
            vector_items = build_ste_index_sql(conn, vector_scores.keys())

    candidates_map: Dict[str, Dict[str, Any]] = {}
    for it in candidates_fts:
        it.pop("bm25", None)
        candidates_map[it["id"]] = it
    for ste_id, item in vector_items.items():
        if ste_id not in candidates_map:
            candidates_map[ste_id] = item

    # CF candidates
    company_history_ids = set([str(x) for x in company_personal_ids]) | set(
        [str(x) for x in purchase_history if isinstance(x, (str, int))]
    )
    cf_scores = cf_candidates(company_history_ids)
    for ste_id in cf_scores.keys():
        if ste_id not in candidates_map:
            fetched = fetch_item_sql(conn, ste_id)
            if fetched:
                candidates_map[ste_id] = fetched

    candidates = list(candidates_map.values())

    # Behavior context
    liked_ids = set(session_feedback.get("liked_ids", []))
    visited_ids = set(session_feedback.get("visited_ids", []))
    long_dwell_ids = set(session_feedback.get("long_dwell_ids", []))
    disliked_ids = set(session_feedback.get("disliked_ids", []))

    boost_tokens, boost_categories = build_behavior_context(
        conn,
        candidates,
        liked_ids,
        visited_ids,
        long_dwell_ids,
    )

    personal_id_set = set(company_history_ids)

    behavior_scores: Dict[str, float] = {}
    for it in candidates:
        behavior_scores[it["id"]] = behavior_score(
            it,
            liked_ids,
            visited_ids,
            long_dwell_ids,
            disliked_ids,
            personal_id_set,
            boost_tokens,
            boost_categories,
            cf_scores,
            config.behavior,
        )

    scored = score_candidates(candidates, bm25_scores, vector_scores, behavior_scores, config)

    for it in scored:
        reasons = personal_reasons(
            it,
            liked_ids,
            visited_ids,
            long_dwell_ids,
            personal_id_set,
            boost_tokens,
            boost_categories,
            cf_scores,
        )
        if reasons:
            it["label"] = "personal"
            it["personal_reason"] = reasons[:3]
        else:
            it["label"] = "personal" if it["id"] in personal_id_set else "search"

    scroll_pattern = "default"
    if liked_ids or visited_ids or long_dwell_ids or cf_scores:
        scroll_pattern = "personalized"

    def interleave(primary: List[Dict], secondary: List[Dict], take_primary: int = 1, take_secondary: int = 1) -> List[Dict]:
        result: List[Dict] = []
        i = j = 0
        while i < len(primary) or j < len(secondary):
            for _ in range(take_primary):
                if i < len(primary):
                    result.append(primary[i])
                    i += 1
            for _ in range(take_secondary):
                if j < len(secondary):
                    result.append(secondary[j])
                    j += 1
        return result

    if scroll_pattern == "personalized":
        primary = [it for it in scored if it.get("label") == "personal"]
        primary_ids = {it["id"] for it in primary}
        secondary = [it for it in scored if it["id"] not in primary_ids]
        scroll_list = interleave(primary, secondary, take_primary=1, take_secondary=1)
        page_items = scroll_list[offset:offset + limit]
    else:
        page_items = scored[offset:offset + limit]

    popular_items: List[Dict[str, Any]] = []
    popular_limit = min(5, limit)
    for ste_id in stats.get("top_ste_ids", [])[:50]:
        item = ste_index.get(ste_id)
        if item:
            if user_query:
                q_tokens = set(normalize_query(user_query).split())
                name = item.get("name", "")
                category = item.get("category", "")
                if q_tokens and not all(t in f"{name} {category}".lower() for t in q_tokens):
                    continue
            it = dict(item)
            it["label"] = "popular"
            popular_items.append(it)
        if len(popular_items) >= popular_limit:
            break

    t_page = time.perf_counter()

    metrics = {
        "total_ms": round((t_page - t0) * 1000, 2),
        "db_ready_ms": round((t_db_ready - t0) * 1000, 2),
        "db_connect_ms": round((t_db_conn - t_db_ready) * 1000, 2),
        "stats_cache_ms": round((t_stats - t_db_conn) * 1000, 2),
        "indices_ms": round((t_indices - t_stats) * 1000, 2),
        "page_build_ms": round((t_page - t_indices) * 1000, 2),
        "candidates": len(candidates),
        "items_returned": len(page_items),
        "cache_hit": cache is not None and cache.get("db_mtime") == db_mtime,
    }

    result = {
        "corrected_query": corrected_query,
        "suggestions": sugg,
        "expanded_queries": expanded,
        "search_strategy": intent,
        "filters": category_filter,
        "ranking_hints": ranking_hints(intent, favorite_categories, personal_names),
        "explanations": explanations(intent, favorite_categories, personal_names, popular_names),
        "scroll_pattern": scroll_pattern,
        "metrics": metrics,
        "items": page_items,
        "personal_items": [i for i in page_items if i.get("label") == "personal"],
        "search_items": [i for i in page_items if i.get("label") == "search"],
        "popular_items": popular_items,
        "customer_companies": stats.get("top_customers", []),
        "supplier_companies": stats.get("top_suppliers", []),
        "page": page,
        "limit": limit,
    }
    conn.close()
    return result


def main() -> None:
    """CLI entrypoint.

    Args:
        None.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="-", help="JSON input path or - for stdin")
    args = parser.parse_args()

    payload = read_json(args.input)
    result = run_agent(payload)
    write_json(result)


if __name__ == "__main__":
    main()
