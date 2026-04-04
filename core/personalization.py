import logging
import os
import pickle
import sqlite3
from typing import Dict, Iterable, List, Set, Tuple

from core.normalization import tokens as norm_tokens
from core.retrieval import fetch_item_sql

logger = logging.getLogger("search.personalization")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CF_PATH = os.environ.get("CF_PATH", os.path.join(ROOT_DIR, "data", "item_cf.pkl"))

_CF_CACHE: Dict[str, List[Tuple[str, float]]] = {}
_CF_LOADED = False


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
) -> Tuple[Set[str], Set[str]]:
    """Collect tokens and categories from liked/visited items.

    Args:
        conn: SQLite connection.
        candidates: Candidate items.
        liked_ids: Liked item ids.
        visited_ids: Visited item ids.
        long_dwell_ids: Long dwell item ids.

    Returns:
        Tuple of (tokens, categories).
    """
    boost_tokens: Set[str] = set()
    boost_categories: Set[str] = set()
    boost_ids = liked_ids | visited_ids | long_dwell_ids
    if not boost_ids:
        return boost_tokens, boost_categories
    candidate_map = {it["id"]: it for it in candidates}
    for bid in boost_ids:
        it = candidate_map.get(bid)
        if not it:
            it = fetch_item_sql(conn, bid)
        if it:
            cat = it.get("category", "")
            if cat:
                boost_categories.add(cat)
            boost_tokens |= set(norm_tokens(it.get("name", "")))
    return boost_tokens, boost_categories


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
