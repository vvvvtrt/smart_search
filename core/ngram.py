import math
from typing import Dict, Iterable, Set

from core.normalization import normalize_query

_CACHE: Dict[str, Set[str]] = {}
_CACHE_LIMIT = 10000


def char_ngrams(text: str, n_min: int = 3, n_max: int = 5) -> Set[str]:
    """Build a set of character n-grams for a text.

    Args:
        text: Input text.
        n_min: Minimum n-gram size.
        n_max: Maximum n-gram size.

    Returns:
        Set of n-grams.
    """
    norm = normalize_query(text)
    if not norm:
        return set()
    norm = f"_{norm.replace(' ', '_')}_"
    grams: Set[str] = set()
    for n in range(n_min, n_max + 1):
        if len(norm) < n:
            continue
        for i in range(len(norm) - n + 1):
            grams.add(norm[i:i + n])
    return grams


def _get_cached(key: str, text: str) -> Set[str]:
    if key in _CACHE:
        return _CACHE[key]
    grams = char_ngrams(text)
    if len(_CACHE) >= _CACHE_LIMIT:
        _CACHE.clear()
    _CACHE[key] = grams
    return grams


def ngram_scores(query: str, items: Iterable[Dict]) -> Dict[str, float]:
    """Compute n-gram cosine similarity scores for candidates.

    Args:
        query: Query text.
        items: Iterable of item dicts with id/name/category.

    Returns:
        Mapping id -> similarity score.
    """
    qgrams = char_ngrams(query)
    if not qgrams:
        return {}
    qlen = math.sqrt(len(qgrams))
    scores: Dict[str, float] = {}
    for it in items:
        item_id = it.get("id")
        if not item_id:
            continue
        text = f"{it.get('name', '')} {it.get('category', '')}".strip()
        igrams = _get_cached(str(item_id), text)
        if not igrams:
            continue
        inter = len(qgrams & igrams)
        if inter == 0:
            continue
        denom = qlen * math.sqrt(len(igrams))
        if denom <= 0:
            continue
        scores[str(item_id)] = inter / denom
    return scores
