from difflib import SequenceMatcher, get_close_matches
from typing import Dict, List

from core.normalization import normalize_query


def detect_intent(query: str, known_categories: List[str]) -> str:
    """Detect query intent.

    Args:
        query: Input query.
        known_categories: Known category list.

    Returns:
        Intent label.
    """
    q = normalize_query(query)
    if not q:
        return "semantic"
    tokens = q.split()
    if any(ch.isdigit() for ch in q) and len(tokens) <= 6:
        return "exact"
    if q in [c.lower() for c in known_categories]:
        return "category"
    if len(tokens) <= 2:
        return "category"
    return "semantic"


def suggest_correction(query: str, vocab: set) -> str | None:
    """Suggest spelling correction.

    Args:
        query: Input query.
        vocab: Vocabulary set.

    Returns:
        Suggested correction or None.
    """
    q = normalize_query(query)
    if not q:
        return None
    if q in vocab:
        return None
    matches = get_close_matches(q, vocab, n=1, cutoff=0.86)
    if matches:
        return matches[0]
    return None


def build_vocab(known_categories: List[str], popular_names: List[str]) -> set:
    """Build vocabulary for correction.

    Args:
        known_categories: Known category list.
        popular_names: Popular item names.

    Returns:
        Vocabulary set.
    """
    vocab = set()
    for c in known_categories:
        if c:
            vocab.add(normalize_query(c))
    for n in popular_names:
        if n:
            vocab.add(normalize_query(n))
    return vocab


def expand_queries(query: str, intent: str, popular_names: List[str], personal_names: List[str], category: str | None) -> List[str]:
    """Expand query with heuristics.

    Args:
        query: Input query.
        intent: Intent label.
        popular_names: Popular item names.
        personal_names: Personal item names.
        category: Inferred category.

    Returns:
        List of expanded queries.
    """
    expansions = []
    q = query.strip()
    if not q:
        return expansions
    if intent == "category":
        expansions.append(f"{q} купить")
        expansions.append(f"{q} оптом")
    if category and category.lower() not in q.lower():
        expansions.append(f"{q} {category}")
    for name in (personal_names + popular_names):
        if q.lower() in name.lower() and name != q:
            expansions.append(name)
        else:
            q_tokens = set(normalize_query(q).split())
            name_tokens = set(normalize_query(name).split())
            if q_tokens and len(q_tokens & name_tokens) >= 1:
                expansions.append(name)
    seen = set()
    result = []
    for e in expansions:
        e = e.strip()
        if e and e not in seen:
            seen.add(e)
            result.append(e)
    return result[:5]


def similar_categories(query: str, categories: List[str], limit: int = 20) -> List[str]:
    """Find categories similar to the query.

    Args:
        query: Input query.
        categories: Category list.
        limit: Max results.

    Returns:
        List of categories.
    """
    q = normalize_query(query)
    if not q:
        return categories[:limit]
    scored = []
    q_tokens = set(q.split())
    for c in categories:
        c_norm = normalize_query(c)
        if not c_norm:
            continue
        c_tokens = set(c_norm.split())
        token_overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))
        ratio = SequenceMatcher(None, q, c_norm).ratio()
        score = 0.7 * ratio + 0.3 * token_overlap
        scored.append((score, c))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [c for _, c in scored[:limit]]


def autocomplete(
    query: str,
    popular_names: List[str],
    personal_names: List[str],
    semantic_terms: List[str],
    popular_by_category: Dict[str, List[str]],
    preferred_categories: List[str],
) -> Dict[str, List[str]]:
    """Build autocomplete suggestions.

    Args:
        query: Input query.
        popular_names: Popular items.
        personal_names: Personal items.
        semantic_terms: Semantic terms.
        popular_by_category: Popular items by category.
        preferred_categories: Preferred categories.

    Returns:
        Suggestions dict.
    """
    q = normalize_query(query)
    q_tokens = set(q.split())

    def pick(source: List[str]) -> List[str]:
        if not q:
            return source[:5]
        filtered = [s for s in source if q in normalize_query(s)]
        return filtered[:5]

    def pick_personal(source: List[str]) -> List[str]:
        if not q:
            return source[:5]
        filtered = [s for s in source if q in normalize_query(s)]
        if filtered:
            return filtered[:5]
        overlap = []
        for s in source:
            s_tokens = set(normalize_query(s).split())
            if q_tokens and len(q_tokens & s_tokens) >= 1:
                overlap.append(s)
        return overlap[:5]

    popular = pick(popular_names)
    if not popular and preferred_categories:
        for c in preferred_categories:
            if c in popular_by_category:
                popular = popular_by_category[c][:5]
                if popular:
                    break
    return {
        "popular": popular,
        "personal": pick_personal(personal_names),
        "semantic": pick(semantic_terms),
    }


def infer_filters(query: str, known_categories: List[str]) -> Dict[str, str]:
    """Infer filters from query.

    Args:
        query: Input query.
        known_categories: Category list.

    Returns:
        Filters dict.
    """
    q = normalize_query(query)
    filters: Dict[str, str] = {}
    for c in known_categories:
        if normalize_query(c) in q:
            filters["category"] = c
            break
    return filters


def ranking_hints(intent: str, favorite_categories: List[str], purchase_history_names: List[str]) -> List[str]:
    """Generate ranking hints.

    Args:
        intent: Intent label.
        favorite_categories: Favorite categories.
        purchase_history_names: Purchase history names.

    Returns:
        List of hints.
    """
    hints = []
    if intent == "exact":
        hints.append("bm25_boost:high")
    if intent == "semantic":
        hints.append("vector_search_boost:high")
    if intent == "category":
        hints.append("category_expansion:enabled")
    for c in favorite_categories[:3]:
        hints.append(f"boost_category:{c}")
    if purchase_history_names:
        hints.append("boost_previously_purchased:enabled")
    return hints


def explanations(intent: str, favorite_categories: List[str], personal_names: List[str], popular_names: List[str]) -> List[str]:
    """Generate explanation strings.

    Args:
        intent: Intent label.
        favorite_categories: Favorite categories.
        personal_names: Personal names.
        popular_names: Popular names.

    Returns:
        List of explanations.
    """
    expl = []
    if personal_names:
        expl.append("Похоже на ваши прошлые закупки")
    if favorite_categories:
        expl.append("Вы часто покупаете эту категорию")
    if popular_names:
        expl.append("Популярный товар среди заказчиков")
    if intent == "semantic":
        expl.append("Запрос размытый — усиливаем смысловой поиск")
    return expl[:5]


def build_personal_names(purchase_history: List, ste_index: Dict[str, Dict], contract_names: Dict[str, str] | None = None) -> List[str]:
    """Build list of personal item names.

    Args:
        purchase_history: Purchase history list.
        ste_index: STE index mapping.
        contract_names: Optional contract names mapping.

    Returns:
        List of names.
    """
    contract_names = contract_names or {}
    names: List[str] = []
    for entry in purchase_history:
        if isinstance(entry, dict):
            ste_id = str(entry.get("ste_id") or entry.get("id") or "").strip()
            name = entry.get("name")
            if not name and ste_id in ste_index:
                name = ste_index[ste_id]["name"]
            if not name and ste_id in contract_names:
                name = contract_names[ste_id]
            if name:
                names.append(name)
        else:
            val = str(entry).strip()
            if val in ste_index:
                names.append(ste_index[val]["name"])
            elif val in contract_names:
                names.append(contract_names[val])
            elif val:
                names.append(val)
    return names
