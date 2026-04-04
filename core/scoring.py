import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from core.normalization import normalize_query, tokens as norm_tokens, category_cluster

logger = logging.getLogger("search.scoring")


@dataclass
class BehaviorWeights:
    like: float = 1.0
    long_dwell: float = 0.6
    visited: float = 0.3
    category_match: float = 0.4
    token_overlap: float = 0.25
    dislike: float = -0.5
    purchase_history: float = 0.35
    cf_boost: float = 0.4


@dataclass
class ScoringWeights:
    alpha: float = 0.15
    beta: float = 0.75
    gamma: float = 0.10


@dataclass
class SearchConfig:
    scoring: ScoringWeights = field(default_factory=ScoringWeights)
    behavior: BehaviorWeights = field(default_factory=BehaviorWeights)
    personal_threshold: float = 0.55
    cluster_softness: float = 0.35
    debug: bool = False


def normalize_scores(values: Dict[str, float]) -> Dict[str, float]:
    """Normalize scores to [0,1].

    Args:
        values: Mapping id -> score.

    Returns:
        Normalized mapping.
    """
    if not values:
        return {}
    vmin = min(values.values())
    vmax = max(values.values())
    if vmax == vmin:
        return {k: 0.0 for k in values}
    return {k: (v - vmin) / (vmax - vmin) for k, v in values.items()}


def behavior_score(
    item: Dict,
    liked_ids: Set[str],
    visited_ids: Set[str],
    long_dwell_ids: Set[str],
    disliked_ids: Set[str],
    personal_id_set: Set[str],
    boost_tokens: Set[str],
    boost_categories: Set[str],
    cf_scores: Dict[str, float],
    weights: BehaviorWeights,
    query_cluster: List[str],
) -> float:
    """Compute raw behavior score for a single item.

    Args:
        item: Candidate item.
        liked_ids: Liked ids.
        visited_ids: Visited ids.
        long_dwell_ids: Long dwell ids.
        disliked_ids: Disliked ids.
        personal_id_set: Purchase history ids.
        boost_tokens: Tokens from liked items.
        boost_categories: Categories from liked items.
        cf_scores: CF similarity scores.
        weights: Behavior weights.

    Returns:
        Raw behavior score.
    """
    raw = 0.0
    item_id = item["id"]
    if item_id in liked_ids:
        raw += weights.like
    if item_id in long_dwell_ids:
        raw += weights.long_dwell
    if item_id in visited_ids:
        raw += weights.visited
    if item_id in disliked_ids:
        raw += weights.dislike
    if item_id in personal_id_set:
        raw += weights.purchase_history
    item_cluster = category_cluster(item.get("category", ""))
    if not query_cluster or not item_cluster:
        cluster_weight = 1.0
    else:
        overlap = len(set(query_cluster) & set(item_cluster))
        cluster_weight = 1.0 if overlap >= 1 else 0.35
    if boost_categories and item.get("category") in boost_categories:
        raw += weights.category_match * cluster_weight
    if boost_tokens:
        name_tokens = set(norm_tokens(item.get("name", "")))
        if name_tokens & boost_tokens:
            raw += weights.token_overlap * cluster_weight
    if item_id in cf_scores:
        raw += weights.cf_boost * cf_scores[item_id] * cluster_weight
    return raw


def normalize_behavior_scores(raw_scores: Dict[str, float], weights: BehaviorWeights) -> Dict[str, float]:
    """Normalize behavior scores to [0,1] based on theoretical min/max.

    Args:
        raw_scores: Raw behavior scores by id.
        weights: Behavior weights.

    Returns:
        Normalized behavior scores.
    """
    max_raw = (
        weights.like
        + weights.long_dwell
        + weights.visited
        + weights.category_match
        + weights.token_overlap
        + weights.purchase_history
        + weights.cf_boost
    )
    min_raw = weights.dislike
    if max_raw == min_raw:
        return {k: 0.0 for k in raw_scores}
    normed = {}
    for k, v in raw_scores.items():
        nv = (v - min_raw) / (max_raw - min_raw)
        if nv < 0:
            nv = 0.0
        if nv > 1:
            nv = 1.0
        normed[k] = nv
    return normed


def score_candidates(
    candidates: List[Dict],
    bm25_scores: Dict[str, float],
    vector_scores: Dict[str, float],
    behavior_scores: Dict[str, float],
    config: SearchConfig,
) -> List[Dict]:
    """Compute final scores and optionally add breakdown.

    Args:
        candidates: Candidate items.
        bm25_scores: BM25 scores.
        vector_scores: Vector similarity scores.
        behavior_scores: Behavior scores.
        config: Search configuration.

    Returns:
        Sorted scored items.
    """
    bm25_norm = normalize_scores(bm25_scores)
    vec_norm = normalize_scores(vector_scores)
    beh_norm = normalize_behavior_scores(behavior_scores, config.behavior)

    scored = []
    for item in candidates:
        item_id = item["id"]
        bm = bm25_norm.get(item_id, 0.0)
        ve = vec_norm.get(item_id, 0.0)
        be = beh_norm.get(item_id, 0.0)
        final = (
            config.scoring.alpha * bm
            + config.scoring.beta * ve
            + config.scoring.gamma * be
        )
        it = dict(item)
        it["score"] = float(final)
        if config.debug:
            it["score_breakdown"] = {
                "bm25": round(bm, 4),
                "vec": round(ve, 4),
                "beh": round(be, 4),
                "final": round(final, 4),
            }
            logger.debug(
                "score_breakdown",
                extra={
                    "id": item_id,
                    "bm25": bm,
                    "vec": ve,
                    "beh": be,
                    "final": final,
                },
            )
        scored.append(it)
    scored.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return scored


def personal_reasons(
    item: Dict,
    liked_ids: Set[str],
    visited_ids: Set[str],
    long_dwell_ids: Set[str],
    personal_id_set: Set[str],
    boost_tokens: Set[str],
    boost_categories: Set[str],
    cf_scores: Dict[str, float],
    query_cluster: List[str],
) -> List[str]:
    """Return list of human-readable reasons for personalization.

    Args:
        item: Candidate item.
        liked_ids: Liked ids.
        visited_ids: Visited ids.
        long_dwell_ids: Long dwell ids.
        personal_id_set: Purchase history ids.
        boost_tokens: Tokens from liked items.
        boost_categories: Categories from liked items.
        cf_scores: CF similarity scores.

    Returns:
        List of reasons.
    """
    reasons: List[str] = []
    item_id = item["id"]
    if item_id in liked_ids:
        reasons.append("лайк")
    if item_id in long_dwell_ids:
        reasons.append("долгое чтение")
    if item_id in visited_ids:
        reasons.append("посещение")
    if item_id in personal_id_set:
        reasons.append("история/профиль компании")
    item_cluster = category_cluster(item.get("category", ""))
    if not query_cluster or not item_cluster:
        cluster_ok = True
    else:
        cluster_ok = len(set(query_cluster) & set(item_cluster)) >= 1
    if cluster_ok and boost_categories and item.get("category") in boost_categories:
        reasons.append("категория интереса")
    if cluster_ok and boost_tokens:
        name_tokens = set(norm_tokens(item.get("name", "")))
        if name_tokens & boost_tokens:
            reasons.append("похоже на понравившееся")
    if cluster_ok and item_id in cf_scores:
        reasons.append("похожие закупки компаний")
    return reasons
