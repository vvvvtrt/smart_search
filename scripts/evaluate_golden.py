import json
import os
from typing import Dict, List, Tuple

from core.normalization import normalize_query
from core.search_agent import run_agent


def _norm(text: str) -> str:
    return normalize_query(text)


def _is_relevant(result_name: str, relevant_names: List[str]) -> bool:
    rn = _norm(result_name)
    for rel in relevant_names:
        reln = _norm(rel)
        if not reln:
            continue
        if reln in rn or rn in reln:
            return True
    return False


def _metrics_at_k(results: List[Dict], relevant_names: List[str], k: int) -> Tuple[float, float, float]:
    top = results[:k]
    if not top:
        return 0.0, 0.0, 0.0
    rel_flags = [_is_relevant(it.get("name", ""), relevant_names) for it in top]
    rel_count = sum(1 for x in rel_flags if x)
    precision = rel_count / k
    recall = rel_count / max(1, len(relevant_names))
    mrr = 0.0
    for i, ok in enumerate(rel_flags, start=1):
        if ok:
            mrr = 1.0 / i
            break
    return precision, recall, mrr


def evaluate(path: str, k_values: List[int]) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    agg = {k: {"precision": 0.0, "recall": 0.0, "mrr": 0.0} for k in k_values}
    ctr = {k: 0.0 for k in k_values}
    total = 0

    for row in data:
        total += 1
        query = row.get("search_query", "")
        customer = row.get("customer", "")
        sphere = row.get("sphere", "")
        relevant = row.get("relevant_products", []) or []

        payload = {
            "user_query": query,
            "company": {"name": customer, "type": "customer"},
            "user_profile": {"favorite_categories": [sphere] if sphere else []},
            "session_feedback": {},
            "limit": max(k_values),
            "page": 1,
        }
        res = run_agent(payload)
        items = res.get("items", [])

        for k in k_values:
            p, r, m = _metrics_at_k(items, relevant, k)
            agg[k]["precision"] += p
            agg[k]["recall"] += r
            agg[k]["mrr"] += m
            ctr[k] += 1.0 if p > 0 else 0.0

    for k in k_values:
        agg[k]["precision"] /= max(1, total)
        agg[k]["recall"] /= max(1, total)
        agg[k]["mrr"] /= max(1, total)
        ctr[k] /= max(1, total)

    return {"total": total, "metrics": agg, "ctr_at_k": ctr}


if __name__ == "__main__":
    dataset = os.environ.get("GOLDEN_PATH", "/Users/mikhail/Downloads/golden_dataset_30.json")
    ks = [5, 10]
    report = evaluate(dataset, ks)
    print(json.dumps(report, ensure_ascii=False, indent=2))
