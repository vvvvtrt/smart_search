#!/usr/bin/env python3
import argparse
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
import logging

from core.normalization import normalize_query, build_query_variants, category_cluster, tokens as norm_tokens
from core.ngram import ngram_scores
from core.personalization import (
    build_behavior_context,
    cf_candidates,
    ensure_industry_profiles,
    get_industry_keywords,
    INDUSTRY_PROFILES,
)
from core.retrieval import (
    DB_PATH,
    ensure_db_ready,
    ensure_lemma_schema,
    compute_contract_stats_sql,
    build_company_profiles_sql,
    load_contract_names_sql,
    build_ste_index_sql,
    top_categories_sql,
    search_like,
    extract_session_tokens,
    weighted_search,
    fetch_item_sql,
    get_company_sectors,
    set_company_sectors,
    company_category_stats,
)
from core.scoring import SearchConfig, behavior_score, category_rerank, personal_reasons, score_candidates
from core.suggestions import (
    autocomplete,
    build_personal_names,
    build_vocab,
    classify_company_sectors,
    detect_intent,
    expand_queries,
    explanations,
    infer_filters,
    ranking_hints,
    similar_categories,
    suggest_correction,
    rewrite_query_variants,
)
from core.utils import read_json, write_json, file_mtime, load_cache, save_cache

logger = logging.getLogger("search.agent")

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
    company_inn = None
    if isinstance(company, dict):
        company_inn = company.get("inn") or company.get("customer_inn") or company.get("supplier_inn")
    company_inn = company_inn or payload.get("company_inn")
    mode = payload.get("mode")
    limit = int(payload.get("limit", 5))
    page = int(payload.get("page", 1))
    offset = max(0, (page - 1) * limit)
    session_feedback = payload.get("session_feedback", {}) or {}
    rec_mode = not normalize_query(user_query)

    config = SearchConfig(debug=os.environ.get("SEARCH_DEBUG", "0") == "1")
    if rec_mode:
        config.scoring.alpha = 0.10
        config.scoring.beta = 0.20
        config.scoring.gamma = 0.70

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
    ensure_industry_profiles(conn)

    cache = load_cache(CACHE_PATH)
    db_mtime = file_mtime(DB_PATH)
    if not cache or cache.get("db_mtime") != db_mtime:
        logger.info("company_classify_start")
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
        # auto-classify sectors for known companies (one-time)
        processed = 0
        saved = 0
        for cname in stats.get("top_customers", [])[:500]:
            processed += 1
            if not get_company_sectors(conn, cname):
                samples = company_category_stats(conn, cname, "customer")
                sectors = classify_company_sectors(cname, samples)
                if sectors:
                    set_company_sectors(conn, cname, sectors)
                    saved += 1
        for cname in stats.get("top_suppliers", [])[:500]:
            processed += 1
            if not get_company_sectors(conn, cname):
                samples = company_category_stats(conn, cname, "supplier")
                sectors = classify_company_sectors(cname, samples)
                if sectors:
                    set_company_sectors(conn, cname, sectors)
                    saved += 1
        logger.info("company_classify_done", extra={"processed": processed, "saved": saved})
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

    if company_name:
        stored = get_company_sectors(conn, company_name)
        if stored:
            favorite_categories = stored
            logger.info("company_sector_loaded", extra={"company": company_name, "sectors": stored})
        elif not favorite_categories:
            samples = company_category_stats(conn, company_name, company_type)
            favorite_categories = classify_company_sectors(company_name, samples)
            if favorite_categories:
                set_company_sectors(conn, company_name, favorite_categories)
                logger.info("company_sector_auto", extra={"company": company_name, "sectors": favorite_categories})
    known_categories = list(dict.fromkeys(favorite_categories + top_categories))

    intent = detect_intent(user_query, known_categories)
    vocab = build_vocab(known_categories, popular_names[:200])
    corrected = suggest_correction(user_query, vocab)
    corrected_query = user_query
    if corrected and corrected != normalize_query(user_query):
        corrected_query = f"Возможно, вы имели в виду: {corrected}"

    category_filter = infer_filters(user_query, known_categories)
    category = category_filter.get("category")
    query_cluster = category_cluster(category or user_query)
    preferred_clusters = [category_cluster(c) for c in favorite_categories if c]
    company_sectors = favorite_categories[:]

    expanded = expand_queries(user_query, intent, popular_names, personal_names, category)
    expanded += rewrite_query_variants(user_query, recent_actions, favorite_categories)
    expanded = list(dict.fromkeys(expanded))[:5]
    preferred_categories = [category] if category else ([] if user_query else favorite_categories)
    sugg = autocomplete(
        user_query,
        popular_names,
        personal_names,
        semantic_terms,
        popular_by_category,
        preferred_categories,
    )

    session_tokens = extract_session_tokens(user_query, session_feedback, conn)
    industry_keywords = get_industry_keywords(str(company_inn), INDUSTRY_PROFILES) if company_inn else []

    # Retrieval: FTS + n-gram similarity
    candidates_fts: List[Dict[str, Any]] = []
    bm25_scores: Dict[str, float] = {}
    vector_scores: Dict[str, float] = {}
    candidates_map: Dict[str, Dict[str, Any]] = {}

    if not rec_mode:
        candidates_fts, bm25_scores = weighted_search(
            user_query,
            session_tokens,
            industry_keywords,
            conn,
            limit=200,
        )
        if not candidates_fts:
            variants = build_query_variants(user_query)
            variants += rewrite_query_variants(user_query, recent_actions, favorite_categories)
            variants = list(dict.fromkeys(variants))[:5]
            if not variants:
                variants = [user_query]
            candidates_fts = search_like(conn, variants, limit=200)
            bm25_scores = {it["id"]: it.get("bm25", 0.0) for it in candidates_fts}

        for it in candidates_fts:
            it.pop("bm25", None)
            candidates_map[it["id"]] = it
        if candidates_map:
            vector_scores = ngram_scores(user_query, candidates_map.values())

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
    if rec_mode and not candidates_map:
        # seed recommendations from company history + popular items
        seed_ids = list(company_history_ids) + list(stats.get("top_ste_ids", [])[:200])
        for ste_id, item in build_ste_index_sql(conn, seed_ids).items():
            candidates_map.setdefault(ste_id, item)

    candidates = list(candidates_map.values())
    q_tokens = [t for t in normalize_query(user_query).split() if t]
    short_query = len(q_tokens) <= 2
    if short_query:
        industry_profile = INDUSTRY_PROFILES.get(str(company_inn), {}) if company_inn else {}
        candidates = category_rerank(candidates, session_feedback, conn, industry_profile)

    # Behavior context
    liked_ids = set(session_feedback.get("liked_ids", []))
    visited_ids = set(session_feedback.get("visited_ids", []))
    long_dwell_ids = set(session_feedback.get("long_dwell_ids", []))
    disliked_ids = set(session_feedback.get("disliked_ids", []))

    recent_action_tokens = set()
    for a in recent_actions:
        if isinstance(a, str):
            recent_action_tokens |= set(norm_tokens(a))

    boost_tokens: Set[str] = set()
    boost_categories: Set[str] = set()
    boost_brands: Set[str] = set()

    _bc = build_behavior_context(
        conn,
        candidates,
        liked_ids,
        visited_ids,
        long_dwell_ids,
    )
    if len(_bc) == 3:
        boost_tokens, boost_categories, boost_brands = _bc

    focus_categories: Set[str] = set()
    if boost_categories:
        focus_categories |= set(boost_categories)
    if short_query and recent_actions:
        for a in recent_actions:
            if isinstance(a, str):
                na = normalize_query(a)
                for cat in top_categories:
                    if na and na in normalize_query(cat):
                        focus_categories.add(cat)

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
            boost_brands,
            recent_action_tokens,
            focus_categories,
            short_query,
            cf_scores,
            config.behavior,
            query_cluster,
            preferred_clusters,
            company_sectors,
        )
    category_boosts = {it["id"]: float(it.get("category_boost", 1.0)) for it in candidates}
    for item_id, score in list(behavior_scores.items()):
        behavior_scores[item_id] = score * category_boosts.get(item_id, 1.0)

    scored = score_candidates(candidates, bm25_scores, vector_scores, behavior_scores, config, category_boosts)

    for it in scored:
        reasons = personal_reasons(
            it,
            liked_ids,
            visited_ids,
            long_dwell_ids,
            personal_id_set,
            boost_tokens,
            boost_categories,
            boost_brands,
            recent_action_tokens,
            focus_categories,
            short_query,
            cf_scores,
            query_cluster,
            preferred_clusters,
            company_sectors,
        )
        # If only company-sector reason and query is present, do not mark as personal
        if normalize_query(user_query) and reasons and set(reasons) == {"сфера деятельности компании"}:
            it["label"] = "search"
        elif reasons:
            it["label"] = "personal"
            it["personal_reason"] = reasons[:3]
        else:
            it["label"] = "personal" if it["id"] in personal_id_set else "search"

    # If too many items look personal for a category/exact query, soften personalization
    if intent in ("category", "exact"):
        personal_ratio = sum(1 for it in scored if it.get("label") == "personal") / max(1, len(scored))
        if personal_ratio >= 0.7:
            for it in scored:
                if it.get("personal_reason"):
                    strong = any(r in it["personal_reason"] for r in ["лайк", "долгое чтение", "история/профиль компании"])
                    if not strong:
                        it["label"] = "search"
                        it.pop("personal_reason", None)

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

    if short_query and focus_categories:
        def _focus_match(it):
            return it.get("category") in focus_categories
        focus_first = [it for it in page_items if _focus_match(it)]
        focus_rest = [it for it in page_items if it not in focus_first]
        page_items = focus_first + focus_rest

    # If query present, prefer items that match company sector clusters
    if normalize_query(user_query) and company_sectors:
        def _sector_match(it):
            from core.suggestions import category_to_sectors
            item_sectors = category_to_sectors(it.get("category", "") + " " + it.get("name", ""))
            return bool(set(item_sectors) & set(company_sectors))
        sector_first = [it for it in page_items if _sector_match(it)]
        sector_rest = [it for it in page_items if it not in sector_first]
        page_items = sector_first + sector_rest

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

    # recommendation categories (from behavior + history)
    rec_categories: List[str] = []
    if boost_categories:
        rec_categories.extend(list(boost_categories))
    for ste_id in list(company_history_ids)[:200]:
        item = ste_index.get(ste_id)
        if item and item.get("category"):
            rec_categories.append(item["category"])
    rec_categories = list(dict.fromkeys(rec_categories))[:8]

    recommendations = scored[: min(8, len(scored))]

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
        "recommendations": recommendations,
        "recommendation_categories": rec_categories,
        "personal_items": [i for i in page_items if i.get("label") == "personal"],
        "search_items": [i for i in page_items if i.get("label") == "search"],
        "popular_items": popular_items,
        "customer_companies": stats.get("top_customers", []),
        "supplier_companies": stats.get("top_suppliers", []),
        "company_sectors": favorite_categories,
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
