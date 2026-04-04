#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher, get_close_matches
from core.model import score_items


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STE_DEFAULT_PATH = os.environ.get(
    "DATA_STE_PATH",
    os.path.join(ROOT_DIR, "data", "СТЕ_20260403.csv"),
)
CONTRACTS_DEFAULT_PATH = os.environ.get(
    "DATA_CONTRACTS_PATH",
    os.path.join(ROOT_DIR, "data", "Контракты_20260403.csv"),
)
CACHE_PATH = os.environ.get(
    "CACHE_PATH",
    os.path.join(ROOT_DIR, ".search_agent_cache.json"),
)
DB_PATH = os.environ.get(
    "DB_PATH",
    os.path.join(ROOT_DIR, "data", "search.db"),
)

_MEMO = {
    "db_mtime": None,
    "top_categories": [],
    "contract_names": {},
    "contract_names_key": None,
}


def ensure_db_ready():
    if not os.path.exists(DB_PATH):
        from core.build_db import build_db
        build_db()
        return
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("SELECT 1 FROM ste_fts LIMIT 1")
        conn.close()
    except Exception:
        from core.build_db import build_db
        build_db()


def read_json(path):
    if not path or path == "-":
        return json.load(os.sys.stdin)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data):
    print(json.dumps(data, ensure_ascii=False, indent=2))


def file_mtime(path):
    try:
        return os.path.getmtime(path)
    except FileNotFoundError:
        return None


def load_cache():
    if not os.path.exists(CACHE_PATH):
        return None
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_cache(payload):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def parse_ste_attributes(attr_text):
    if not attr_text:
        return {}
    parts = [p for p in attr_text.split(";") if ":" in p]
    attrs = {}
    for p in parts:
        key, val = p.split(":", 1)
        key = key.strip()
        val = val.strip()
        if key and val:
            attrs[key] = val
    return attrs


def stream_csv(path):
    with open(path, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter=";", quotechar='"')
        for row in reader:
            if row:
                yield row


def compute_contract_stats(contracts_path, top_n=5000, top_companies=500):
    counter = Counter()
    customers = Counter()
    suppliers = Counter()
    total = 0
    for row in stream_csv(contracts_path):
        if len(row) < 3:
            continue
        ste_id = row[2].strip()
        if ste_id:
            counter[ste_id] += 1
            total += 1
        if len(row) > 6 and row[6].strip():
            customers[row[6].strip()] += 1
        if len(row) > 9 and row[9].strip():
            suppliers[row[9].strip()] += 1
    top = counter.most_common(top_n)
    return {
        "total_contract_rows": total,
        "top_ste_ids": [sid for sid, _ in top],
        "top_ste_counts": {sid: cnt for sid, cnt in top},
        "top_customers": [n for n, _ in customers.most_common(top_companies)],
        "top_suppliers": [n for n, _ in suppliers.most_common(top_companies)],
    }


def load_contract_names(contracts_path, ste_ids, limit_per_id=1):
    names = {}
    remaining = set(ste_ids)
    if not remaining:
        return names
    for row in stream_csv(contracts_path):
        if len(row) < 3:
            continue
        ste_id = row[2].strip()
        if ste_id in remaining:
            item_name = row[0].strip().strip('"')
            if item_name:
                names[ste_id] = item_name
                remaining.discard(ste_id)
            if not remaining:
                break
    return names


def chunked(values, size=900):
    buf = []
    for v in values:
        buf.append(v)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def compute_contract_stats_sql(conn, top_n=5000, top_companies=500):
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


def load_contract_names_sql(conn, ste_ids):
    names = {}
    if not ste_ids:
        return names
    for batch in chunked(ste_ids):
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


def build_ste_index_sql(conn, needed_ids):
    index = {}
    if not needed_ids:
        return index
    for batch in chunked(list(needed_ids)):
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
                "attributes": parse_ste_attributes(attributes),
            }
    return index


def top_categories_sql(conn, limit=2000):
    rows = conn.execute(
        "SELECT category, COUNT(1) as cnt FROM ste WHERE category != '' "
        "GROUP BY category ORDER BY cnt DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [c for c, _ in rows]


def build_company_profiles_sql(conn, companies, kind="customer", limit_ids=200):
    company_to_counts = {c: Counter() for c in companies}
    if not company_to_counts:
        return {}
    col = "customer_name" if kind == "customer" else "supplier_name"
    for name in companies:
        rows = conn.execute(
            f"SELECT ste_id, COUNT(1) as cnt FROM contracts WHERE {col} = ? GROUP BY ste_id ORDER BY cnt DESC LIMIT ?",
            (name, limit_ids),
        ).fetchall()
        for ste_id, cnt in rows:
            company_to_counts[name][ste_id] = cnt
    profiles = {}
    for name, cnt in company_to_counts.items():
        profiles[name] = [sid for sid, _ in cnt.most_common(limit_ids)]
    return profiles


def match_query(name, category, query_tokens):
    if not query_tokens:
        return False
    hay = f"{name} {category}".lower()
    return all(t in hay for t in query_tokens)


def search_catalog_sql(conn, query, limit=50, offset=0):
    q = normalize_query(query)
    if not q:
        return []
    rows = conn.execute(
        "SELECT s.id, s.name, s.category, s.attributes "
        "FROM ste_fts f JOIN ste s ON s.rowid = f.rowid "
        "WHERE ste_fts MATCH ? LIMIT ? OFFSET ?",
        (q, limit, offset),
    ).fetchall()
    if not rows:
        tokens = [t for t in q.split() if t]
        if tokens:
            where = " AND ".join(["(name LIKE ? OR category LIKE ?)"] * len(tokens))
            params = []
            for t in tokens:
                like = f"%{t}%"
                params.extend([like, like])
            params.extend([limit, offset])
            rows = conn.execute(
                f"SELECT id, name, category, attributes FROM ste WHERE {where} LIMIT ? OFFSET ?",
                params,
            ).fetchall()
    results = []
    for ste_id, name, category, attributes in rows:
        results.append({
            "id": str(ste_id),
            "name": name or "",
            "category": category or "",
            "attributes": parse_ste_attributes(attributes),
        })
    return results


def fetch_item_sql(conn, item_id):
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
        "attributes": parse_ste_attributes(attributes),
    }


def build_ste_index(ste_path, needed_ids):
    index = {}
    categories = Counter()
    for row in stream_csv(ste_path):
        if len(row) < 3:
            continue
        ste_id = row[0].strip()
        if ste_id in needed_ids:
            name = row[1].strip() if len(row) > 1 else ""
            category = row[2].strip() if len(row) > 2 else ""
            attrs = parse_ste_attributes(row[3]) if len(row) > 3 else {}
            index[ste_id] = {
                "id": ste_id,
                "name": name,
                "category": category,
                "attributes": attrs,
            }
        if len(row) > 2 and row[2].strip():
            categories[row[2].strip()] += 1
    top_categories = [c for c, _ in categories.most_common(2000)]
    return index, top_categories


def normalize_query(q):
    return " ".join(q.strip().lower().split())


def detect_intent(query, known_categories):
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


def suggest_correction(query, vocab):
    q = normalize_query(query)
    if not q:
        return None
    if q in vocab:
        return None
    matches = get_close_matches(q, vocab, n=1, cutoff=0.86)
    if matches:
        return matches[0]
    return None


def build_vocab(known_categories, popular_names):
    vocab = set()
    for c in known_categories:
        if c:
            vocab.add(normalize_query(c))
    for n in popular_names:
        if n:
            vocab.add(normalize_query(n))
    return vocab


def expand_queries(query, intent, popular_names, personal_names, category):
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


def similar_categories(query, categories, limit=20):
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


def autocomplete(query, popular_names, personal_names, semantic_terms, popular_by_category, preferred_categories):
    q = normalize_query(query)
    q_tokens = set(q.split())
    def pick(source):
        if not q:
            return source[:5]
        filtered = [s for s in source if q in normalize_query(s)]
        return filtered[:5]
    def pick_personal(source):
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


def infer_filters(query, known_categories):
    q = normalize_query(query)
    filters = {}
    for c in known_categories:
        if normalize_query(c) in q:
            filters["category"] = c
            break
    return filters


def ranking_hints(intent, favorite_categories, purchase_history_names):
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


def explanations(intent, favorite_categories, personal_names, popular_names):
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


def build_personal_names(purchase_history, ste_index, contract_names=None):
    contract_names = contract_names or {}
    names = []
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


def run_agent(payload, ste_path=STE_DEFAULT_PATH, contracts_path=CONTRACTS_DEFAULT_PATH):
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

    ensure_db_ready()
    t_db_ready = time.perf_counter()
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA cache_size=200000;")
        conn.execute("PRAGMA mmap_size=268435456;")
    except Exception:
        pass
    t_db_conn = time.perf_counter()

    cache = load_cache()
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
        save_cache(cache)
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

    needed_ids = set(stats.get("top_ste_ids", []))
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

    popular_names = []
    popular_by_category = defaultdict(list)
    for ste_id in stats.get("top_ste_ids", [])[:200]:
        item = ste_index.get(ste_id)
        if item and item["name"]:
            popular_names.append(item["name"])
            if item.get("category"):
                popular_by_category[item["category"]].append(item["name"])
        elif ste_id in contract_names:
            popular_names.append(contract_names[ste_id])

    personal_names = build_personal_names(purchase_history, ste_index, contract_names)
    company_personal_ids = []
    if company_name:
        profile_map = customer_profiles if company_type == "customer" else supplier_profiles
        company_personal_ids = profile_map.get(company_name, [])

    semantic_terms = []
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

    candidates = search_catalog_sql(conn, user_query, limit=200, offset=0)
    personal_id_set = set([str(x) for x in company_personal_ids] + [str(x) for x in purchase_history if isinstance(x, (str, int))])
    labeled_personal = []
    labeled_search = []
    labeled_popular = []
    for item in candidates:
        label = "search"
        if item["id"] in personal_id_set:
            label = "personal"
        item["label"] = label
        if label == "personal":
            labeled_personal.append(item)
        else:
            labeled_search.append(item)

    popular_limit = min(5, limit)
    for ste_id in stats.get("top_ste_ids", [])[:50]:
        item = ste_index.get(ste_id)
        if item:
            if user_query:
                q_tokens = set(normalize_query(user_query).split())
                name = item.get("name", "")
                category = item.get("category", "")
                if q_tokens and not match_query(name, category, q_tokens):
                    continue
            item = dict(item)
            item["label"] = "popular"
            labeled_popular.append(item)
        if len(labeled_popular) >= popular_limit:
            break

    reranked = None
    if candidates:
        features = []
        q = normalize_query(user_query)
        q_tokens = set(q.split())
        for item in candidates:
            name = item.get("name", "")
            category = item.get("category", "")
            text_score = len(q_tokens & set(normalize_query(name + " " + category).split()))
            features.append({
                "query_text": q,
                "item_id": item.get("id", ""),
                "item_name": name,
                "item_category": category,
                "company_name": company_name or "",
                "text_match_score": text_score,
                "is_personal": 1 if item["id"] in personal_id_set else 0,
            })
        reranked = score_items(candidates, features)
    t_rerank = time.perf_counter()

    # session-based similarity boost (likes / visits / long dwell)
    liked_ids = set(session_feedback.get("liked_ids", []))
    visited_ids = set(session_feedback.get("visited_ids", []))
    long_dwell_ids = set(session_feedback.get("long_dwell_ids", []))
    boost_ids = liked_ids | visited_ids | long_dwell_ids
    boost_tokens = set()
    boost_categories = set()
    if boost_ids:
        for bid in boost_ids:
            for it in candidates:
                if it["id"] == bid:
                    boost_categories.add(it.get("category", ""))
                    boost_tokens |= set(normalize_query(it.get("name", "")).split())
                    break

    def is_personal_hit(it):
        if it["id"] in personal_id_set:
            return True
        if it["id"] in boost_ids:
            return True
        if boost_categories and it.get("category") in boost_categories:
            return True
        if boost_tokens:
            name_tokens = set(normalize_query(it.get("name", "")).split())
            if name_tokens & boost_tokens:
                return True
        return False

    scroll_pattern = "default"
    if liked_ids or visited_ids or long_dwell_ids:
        scroll_pattern = "personalized"

    def interleave(primary, secondary, take_primary=2, take_secondary=1):
        result = []
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

    if reranked:
        liked = set(session_feedback.get("liked_ids", []))
        disliked = set(session_feedback.get("disliked_ids", []))
        for it in reranked:
            if it["id"] in liked:
                it["score"] += 0.2
            if it["id"] in disliked:
                it["score"] -= 0.2
            if boost_categories and it.get("category") in boost_categories:
                it["score"] += 0.1
            if boost_tokens:
                name_tokens = set(normalize_query(it.get("name", "")).split())
                if name_tokens & boost_tokens:
                    it["score"] += 0.05
        reranked.sort(key=lambda x: x.get("score", 0), reverse=True)
        if scroll_pattern == "personalized":
            primary = [it for it in reranked if is_personal_hit(it)]
            primary_ids = {it["id"] for it in primary}
            secondary = [it for it in reranked if it["id"] not in primary_ids]
            scroll_list = interleave(primary, secondary, take_primary=1, take_secondary=1)
            page_items = scroll_list[offset:offset + limit]
        else:
            page_items = reranked[offset:offset + limit]
    else:
        page_items = (labeled_personal + labeled_search)[offset:offset + limit]
    t_page = time.perf_counter()

    metrics = {
        "total_ms": round((t_page - t0) * 1000, 2),
        "db_ready_ms": round((t_db_ready - t0) * 1000, 2),
        "db_connect_ms": round((t_db_conn - t_db_ready) * 1000, 2),
        "stats_cache_ms": round((t_stats - t_db_conn) * 1000, 2),
        "indices_ms": round((t_indices - t_stats) * 1000, 2),
        "rerank_ms": round((t_rerank - t_indices) * 1000, 2),
        "page_build_ms": round((t_page - t_rerank) * 1000, 2),
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
        "popular_items": labeled_popular,
        "customer_companies": stats.get("top_customers", []),
        "supplier_companies": stats.get("top_suppliers", []),
        "page": page,
        "limit": limit,
    }
    conn.close()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="-", help="JSON input path or - for stdin")
    parser.add_argument("--ste", default=STE_DEFAULT_PATH)
    parser.add_argument("--contracts", default=CONTRACTS_DEFAULT_PATH)
    args = parser.parse_args()

    payload = read_json(args.input)
    result = run_agent(payload, ste_path=args.ste, contracts_path=args.contracts)
    write_json(result)


if __name__ == "__main__":
    main()
