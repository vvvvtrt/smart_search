import re
from functools import lru_cache
from typing import Iterable, List, Dict

try:
    import pymorphy3
except ImportError:  # pragma: no cover - handled by runtime dependency
    pymorphy3 = None


TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яёЁ]+")

_BRAND_VARIANTS: Dict[str, List[str]] = {
    "asus": ["asus", "асус"],
    "acer": ["acer", "асер"],
    "lenovo": ["lenovo", "леново"],
    "hp": ["hp", "хп"],
    "huawei": ["huawei", "хуавей"],
    "samsung": ["samsung", "самсунг"],
    "xiaomi": ["xiaomi", "ксиаоми", "сяоми"],
    "apple": ["apple", "эппл"],
    "iphone": ["iphone", "айфон"],
    "macbook": ["macbook", "макбук"],
    "dell": ["dell", "делл"],
    "msi": ["msi", "мси"],
    "sony": ["sony", "сони"],
    "lg": ["lg", "лж", "элджи"],
    "yandex": ["yandex", "яндекс"],
}

_EN_RU_LAYOUT = str.maketrans(
    "qwertyuiop[]asdfghjkl;'zxcvbnm,./",
    "йцукенгшщзхъфывапролджэячсмитьбю.",
)
_RU_EN_LAYOUT = str.maketrans(
    "йцукенгшщзхъфывапролджэячсмитьбю.",
    "qwertyuiop[]asdfghjkl;'zxcvbnm,./",
)

_LAT_CYR = {
    "a": "а", "b": "б", "v": "в", "g": "г", "d": "д", "e": "е",
    "z": "з", "i": "и", "k": "к", "l": "л", "m": "м", "n": "н",
    "o": "о", "p": "п", "r": "р", "s": "с", "t": "т", "u": "у",
    "f": "ф", "h": "х", "y": "й",
}


def normalize_query(text: str) -> str:
    """Normalize whitespace and lowercase.

    Args:
        text: Input string.

    Returns:
        Normalized string.
    """
    return " ".join((text or "").strip().lower().split())


def normalize_fts_query(text: str) -> str:
    """Keep only alnum tokens for safe FTS queries.

    Args:
        text: Input string.

    Returns:
        FTS-safe string.
    """
    if not text:
        return ""
    tokens = TOKEN_RE.findall(text.lower())
    return " ".join(tokens)


def tokens(text: str) -> List[str]:
    """Tokenize text into alnum tokens.

    Args:
        text: Input string.

    Returns:
        List of tokens.
    """
    if not text:
        return []
    return TOKEN_RE.findall(text.lower())


@lru_cache(maxsize=50000)
def _lemma_token(token: str) -> str:
    """Return lemma for a single token.

    Args:
        token: Input token.

    Returns:
        Lemma for token.
    """
    if not token:
        return token
    if pymorphy3 is None:
        return token
    morph = _get_morph()
    parsed = morph.parse(token)
    if not parsed:
        return token
    return parsed[0].normal_form


def _get_morph():
    """Singleton morph analyzer.

    Returns:
        Pymorphy3 analyzer instance.
    """
    if pymorphy3 is None:
        raise RuntimeError("pymorphy3 is required but not installed")
    if not hasattr(_get_morph, "_inst"):
        _get_morph._inst = pymorphy3.MorphAnalyzer()
    return _get_morph._inst


def lemmatize_text(text: str) -> str:
    """Lemmatize all tokens in text.

    Args:
        text: Input string.

    Returns:
        Lemmatized string.
    """
    toks = tokens(text)
    if not toks:
        return ""
    lemmas = [_lemma_token(t) for t in toks]
    return " ".join(lemmas)


def category_cluster(text: str) -> List[str]:
    """Derive a coarse cluster token list from text/category.

    Args:
        text: Category or query text.

    Returns:
        List of cluster tokens (up to 3).
    """
    lem = lemmatize_text(text)
    toks = tokens(lem)
    if not toks:
        return []
    return toks[:3]


def swap_layout(text: str) -> str:
    """Swap keyboard layout between EN and RU.

    Args:
        text: Input string.

    Returns:
        Layout-swapped string.
    """
    if not text:
        return text
    latin = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    cyr = sum(1 for ch in text if ("а" <= ch.lower() <= "я") or ch.lower() == "ё")
    if latin >= cyr:
        return text.translate(_EN_RU_LAYOUT)
    return text.translate(_RU_EN_LAYOUT)


def translit_lat_to_cyr(text: str) -> str:
    """Rudimentary transliteration from latin to cyrillic.

    Args:
        text: Input string.

    Returns:
        Transliteration string.
    """
    if not text:
        return text
    out = []
    for ch in text:
        low = ch.lower()
        if low in _LAT_CYR:
            mapped = _LAT_CYR[low]
            out.append(mapped.upper() if ch.isupper() else mapped)
        else:
            out.append(ch)
    return "".join(out)


def expand_brand_variants(toks: Iterable[str]) -> List[str]:
    """Expand known brand variants for each token.

    Args:
        toks: Tokens list.

    Returns:
        Expanded variants list.
    """
    if not toks:
        return []
    variants = []
    for t in toks:
        base = normalize_query(t)
        found = None
        for vals in _BRAND_VARIANTS.values():
            if base in vals:
                found = vals
                break
        variants.append(found or [t])
    results = [""]
    for opts in variants:
        next_res = []
        for r in results:
            for o in opts:
                next_res.append((r + " " + o).strip())
        results = next_res[:10]
    return results


def build_query_variants(query: str, max_variants: int = 5) -> List[str]:
    """Build query variants with layout swap, translit, brand and lemma forms.

    Args:
        query: Input query.
        max_variants: Maximum number of variants.

    Returns:
        List of query variants.
    """
    q = normalize_query(query)
    if not q:
        return []
    variants = [q]
    layout_swapped = normalize_query(swap_layout(q))
    if layout_swapped and layout_swapped not in variants:
        variants.append(layout_swapped)
    translit = normalize_query(translit_lat_to_cyr(q))
    if translit and translit not in variants:
        variants.append(translit)

    toks = q.split()
    for v in expand_brand_variants(toks):
        v = normalize_query(v)
        if v and v not in variants:
            variants.append(v)

    lemma = normalize_query(lemmatize_text(q))
    if lemma and lemma not in variants:
        variants.append(lemma)

    # Add lemma forms for swapped/translit variants
    for base in [layout_swapped, translit]:
        if base:
            lv = normalize_query(lemmatize_text(base))
            if lv and lv not in variants:
                variants.append(lv)

    dedup = []
    seen = set()
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            dedup.append(v)
        if len(dedup) >= max_variants:
            break
    return dedup
