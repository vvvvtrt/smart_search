import logging
import os
import pickle
import sqlite3
from typing import Dict, Iterable, List, Set, Tuple

from core.normalization import lemmatize_text, tokens as norm_tokens
from core.retrieval import fetch_item_sql

logger = logging.getLogger("search.personalization")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CF_PATH = os.environ.get("CF_PATH", os.path.join(ROOT_DIR, "data", "item_cf.pkl"))

_CF_CACHE: Dict[str, List[Tuple[str, float]]] = {}
_CF_LOADED = False
INDUSTRY_PROFILES: Dict[str, Dict[str, float]] = {}
_INDUSTRY_BUILT = False


def load_cf_map() -> Dict[str, List[Tuple[str, float]]]:
    """Load item-item CF map from disk.

    Args:
        None.

    Returns:
        CF map dict.
    """
    global _CF_LOADED
    if _CF_LOADED:
        return _CF_CACHE
    if not os.path.exists(CF_PATH):
        logger.warning("cf_missing", extra={"path": CF_PATH})
        _CF_LOADED = True
        return _CF_CACHE
    try:
        with open(CF_PATH, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            _CF_CACHE.update(data)
    except Exception as exc:
        logger.warning("cf_load_failed", extra={"error": str(exc)})
    _CF_LOADED = True
    return _CF_CACHE


def build_behavior_context(
    conn: sqlite3.Connection,
    candidates: Iterable[Dict],
    liked_ids: Set[str],
    visited_ids: Set[str],
    long_dwell_ids: Set[str],
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Collect tokens, categories, and brand tokens from liked/visited items.

    Args:
        conn: SQLite connection.
        candidates: Candidate items.
        liked_ids: Liked item ids.
        visited_ids: Visited item ids.
        long_dwell_ids: Long dwell item ids.

    Returns:
        Tuple of (tokens, categories, brand_tokens).
    """
    boost_tokens: Set[str] = set()
    boost_categories: Set[str] = set()
    boost_brands: Set[str] = set()
    boost_ids = liked_ids | visited_ids | long_dwell_ids
    if not boost_ids:
        return boost_tokens, boost_categories, boost_brands
    candidate_map = {it["id"]: it for it in candidates}
    for bid in boost_ids:
        it = candidate_map.get(bid)
        if not it:
            it = fetch_item_sql(conn, bid)
        if it:
            cat = it.get("category", "")
            if cat:
                boost_categories.add(cat)
            toks = set(norm_tokens(it.get("name", "")))
            boost_tokens |= toks
            for t in toks:
                if t in {"asus","acer","lenovo","hp","huawei","samsung","xiaomi","apple","iphone","macbook","dell","msi","sony","lg"}:
                    boost_brands.add(t)
    return boost_tokens, boost_categories, boost_brands


def cf_candidates(purchased_ids: Iterable[str], top_k_per_item: int = 30) -> Dict[str, float]:
    """Return CF-suggested candidates and max similarity score.

    Args:
        purchased_ids: Iterable of purchased STE ids.
        top_k_per_item: Top-K similar per item.

    Returns:
        Mapping ste_id -> score.
    """
    cf_map = load_cf_map()
    if not cf_map:
        return {}
    scores: Dict[str, float] = {}
    for pid in purchased_ids:
        for sid, score in cf_map.get(str(pid), [])[:top_k_per_item]:
            if sid == pid:
                continue
            if score > scores.get(sid, 0.0):
                scores[sid] = float(score)
    return scores


def build_industry_profiles(contracts_df) -> Dict[str, Dict[str, float]]:
    """Build industry profiles by customer INN."""
    profiles: Dict[str, Dict[str, float]] = {}
    if contracts_df is None or contracts_df.empty:
        return profiles
    if "customer_inn" not in contracts_df.columns or "category" not in contracts_df.columns:
        return profiles
    if "cost" not in contracts_df.columns:
        return profiles

    df = contracts_df.copy()
    df["customer_inn"] = df["customer_inn"].astype(str)
    df["category"] = df["category"].astype(str)
    df["cost"] = df["cost"].fillna(0)
    try:
        import pandas as pd  # type: ignore
        df["cost"] = pd.to_numeric(df["cost"], errors="coerce").fillna(0.0)
    except Exception:
        def _to_float(val):
            try:
                return float(val)
            except Exception:
                return 0.0
        df["cost"] = df["cost"].apply(_to_float)

    df = df[(df["customer_inn"].str.strip() != "") & (df["category"].str.strip() != "")]
    if df.empty:
        return profiles

    counts = df.groupby("customer_inn").size()
    for inn, group in df.groupby("customer_inn"):
        inn_key = str(inn)
        if int(counts.get(inn, 0)) < 3:
            profiles[inn_key] = {}
            continue
        cat_weights = group.groupby("category")["cost"].sum()
        total = float(cat_weights.sum())
        if total <= 0:
            profiles[inn_key] = {}
            continue
        cat_weights = cat_weights / total
        profiles[inn_key] = {
            cat: float(w)
            for cat, w in cat_weights.items()
            if w >= 0.05
        }
    return profiles


def ensure_industry_profiles(conn: sqlite3.Connection) -> Dict[str, Dict[str, float]]:
    """Ensure industry profiles are built once per process."""
    global _INDUSTRY_BUILT, INDUSTRY_PROFILES
    if _INDUSTRY_BUILT:
        return INDUSTRY_PROFILES
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        logger.warning("industry_profiles_pandas_missing", extra={"error": str(exc)})
        _INDUSTRY_BUILT = True
        return INDUSTRY_PROFILES

    cols = [row[1] for row in conn.execute("PRAGMA table_info(contracts)").fetchall()]
    cost_col = "cost" if "cost" in cols else ("price" if "price" in cols else None)
    if cost_col is None:
        _INDUSTRY_BUILT = True
        return INDUSTRY_PROFILES
    has_category = "category" in cols
    if has_category:
        query = (
            "SELECT customer_inn, "
            f"{cost_col} as cost, "
            "category "
            "FROM contracts WHERE customer_inn != '' AND category != ''"
        )
    else:
        query = (
            "SELECT c.customer_inn, "
            f"c.{cost_col} as cost, "
            "s.category as category "
            "FROM contracts c JOIN ste s ON s.id = c.ste_id "
            "WHERE c.customer_inn != '' AND s.category != ''"
        )
    try:
        contracts_df = pd.read_sql_query(query, conn)
    except Exception as exc:
        logger.warning("industry_profiles_query_failed", extra={"error": str(exc)})
        _INDUSTRY_BUILT = True
        return INDUSTRY_PROFILES

    profiles = build_industry_profiles(contracts_df)
    INDUSTRY_PROFILES.clear()
    INDUSTRY_PROFILES.update(profiles)
    _INDUSTRY_BUILT = True
    return INDUSTRY_PROFILES


def get_industry_keywords(company_inn: str, industry_profiles: Dict[str, Dict[str, float]]) -> List[str]:
    """Return lemmatized top industry keywords for a company."""
    if not company_inn:
        return []
    profile = industry_profiles.get(str(company_inn)) or {}
    if not profile:
        return []
    top_cats = sorted(profile.items(), key=lambda x: (-x[1], x[0]))[:2]
    tokens: List[str] = []
    seen: Set[str] = set()
    for cat, _ in top_cats:
        lem = lemmatize_text(cat)
        for tok in norm_tokens(lem):
            if tok and tok not in seen:
                tokens.append(tok)
                seen.add(tok)
    return tokens
