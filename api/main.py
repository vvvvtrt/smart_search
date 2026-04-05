from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from collections import deque
import math

import os
import logging
import sqlite3
from core.search_agent import run_agent
from core.personalization import ensure_industry_profiles
from core.retrieval import fetch_item_sql, DB_PATH, ensure_db_ready, get_company_sectors, set_company_sectors
from core.build_db import build_db


class UserProfile(BaseModel):
    favorite_categories: List[str] = Field(default_factory=list)
    purchase_history: List[Any] = Field(default_factory=list)
    recent_actions: List[str] = Field(default_factory=list)


class Company(BaseModel):
    type: str = "customer"
    name: str = ""


class SearchPayload(BaseModel):
    user_query: str
    user_profile: UserProfile = Field(default_factory=UserProfile)
    company: Optional[Company] = None
    company_inn: Optional[str] = None
    search_context: Optional[Dict[str, Any]] = None
    page: int = 1
    limit: int = 5
    session_feedback: Optional[Dict[str, Any]] = None


class CompanySectorPayload(BaseModel):
    company_name: str
    sectors: List[str] = Field(default_factory=list)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_query": "флешка 16 гб",
                    "user_profile": {
                        "favorite_categories": [
                            "Usb-накопители твердотельные (флеш-драйвы)"
                        ],
                        "purchase_history": ["1222958"],
                        "recent_actions": ["флеш накопитель smartbuy"],
                    },
                    "company": {"type": "customer", "name": "Государственное бюджетное учреждение здравоохранения Пермского края \"Осинская центральная районная больница\""},
                    "page": 1,
                    "limit": 5,
                    "session_feedback": {"liked_ids": ["1222958"], "disliked_ids": []},
                }
            ]
        }
    }


app = FastAPI(title="Tender Search Agent")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.on_event("startup")
def ensure_db():
    if not os.path.exists(DB_PATH):
        return
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("SELECT 1 FROM ste_fts LIMIT 1")
        ensure_industry_profiles(conn)
        conn.close()
    except Exception:
        build_db()


@app.post("/search")
def search(payload: SearchPayload):
    result = run_agent(payload.model_dump())
    _log_session(payload.model_dump(), result)
    return result


@app.get("/company/sector")
def company_sector(name: str):
    ensure_db_ready()
    conn = sqlite3.connect(DB_PATH)
    data = get_company_sectors(conn, name)
    conn.close()
    return {"company_name": name, "sectors": data}


@app.post("/company/sector")
def set_company_sector(payload: CompanySectorPayload):
    ensure_db_ready()
    conn = sqlite3.connect(DB_PATH)
    set_company_sectors(conn, payload.company_name, payload.sectors)
    data = get_company_sectors(conn, payload.company_name)
    conn.close()
    return {"company_name": payload.company_name, "sectors": data}


@app.get("/companies")
def companies(kind: str = "customer"):
    ensure_db_ready()
    conn = sqlite3.connect(DB_PATH)
    if kind == "supplier":
        rows = conn.execute(
            "SELECT supplier_inn, supplier_name, COUNT(1) as cnt "
            "FROM contracts WHERE supplier_inn != '' AND supplier_name != '' "
            "GROUP BY supplier_inn, supplier_name ORDER BY cnt DESC LIMIT 500"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT customer_inn, customer_name, COUNT(1) as cnt "
            "FROM contracts WHERE customer_inn != '' AND customer_name != '' "
            "GROUP BY customer_inn, customer_name ORDER BY cnt DESC LIMIT 500"
        ).fetchall()
    conn.close()
    companies = [{"inn": str(inn), "name": name} for inn, name, _ in rows if inn and name]
    return {"companies": companies}


@app.get("/item/{item_id}")
def item(item_id: str):
    ensure_db_ready()
    conn = sqlite3.connect(DB_PATH)
    data = fetch_item_sql(conn, item_id)
    conn.close()
    return data or {"error": "not_found"}


app.mount("/", StaticFiles(directory="web", html=True), name="web")


# --- Quality metrics ---
_SESSION_LOG = deque(maxlen=1000)


def _log_session(payload: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Log session data for quality metrics.

    Args:
        payload: Request payload.
        result: Search result.

    Returns:
        None.
    """
    items = [it.get("id") for it in result.get("items", []) if it.get("id")]
    feedback = payload.get("session_feedback", {}) or {}
    _SESSION_LOG.append({
        "items": items,
        "liked_ids": set(feedback.get("liked_ids", [])),
        "visited_ids": set(feedback.get("visited_ids", [])),
        "long_dwell_ids": set(feedback.get("long_dwell_ids", [])),
    })


def _relevance(item_id: str, liked: set, visited: set, dwell: set) -> float:
    """Compute relevance score for an item.

    Args:
        item_id: Item id.
        liked: Liked ids.
        visited: Visited ids.
        dwell: Long dwell ids.

    Returns:
        Relevance score.
    """
    if item_id in liked:
        return 1.0
    if item_id in dwell:
        return 0.7
    if item_id in visited:
        return 0.3
    return 0.0


def _dcg_at_k(items: List[str], liked: set, visited: set, dwell: set, k: int = 10) -> float:
    """Compute DCG@k.

    Args:
        items: Ranked item ids.
        liked: Liked ids.
        visited: Visited ids.
        dwell: Long dwell ids.
        k: Cutoff.

    Returns:
        DCG value.
    """
    dcg = 0.0
    for i, item_id in enumerate(items[:k], start=1):
        rel = _relevance(item_id, liked, visited, dwell)
        if rel > 0:
            dcg += (2 ** rel - 1) / math.log2(i + 1)
    return dcg


@app.get("/metrics/quality")
def quality_metrics():
    """Return aggregated quality metrics.

    Args:
        None.

    Returns:
        Metrics dict.
    """
    if not _SESSION_LOG:
        return {
            "ndcg_at_10": 0.0,
            "mrr": 0.0,
            "ctr_by_position": {},
            "session_success_rate": 0.0,
        }
    ndcgs = []
    mrrs = []
    ctr_clicks = {}
    ctr_impr = {}
    success = 0
    for s in _SESSION_LOG:
        items = s["items"]
        liked = s["liked_ids"]
        visited = s["visited_ids"]
        dwell = s["long_dwell_ids"]
        if liked or dwell:
            success += 1
        dcg = _dcg_at_k(items, liked, visited, dwell, k=10)
        rels = sorted([_relevance(i, liked, visited, dwell) for i in items[:10]], reverse=True)
        ideal = 0.0
        for idx, rel in enumerate(rels, start=1):
            if rel > 0:
                ideal += (2 ** rel - 1) / math.log2(idx + 1)
        ndcgs.append(dcg / ideal if ideal > 0 else 0.0)
        # MRR
        rr = 0.0
        for idx, item_id in enumerate(items, start=1):
            if _relevance(item_id, liked, visited, dwell) > 0:
                rr = 1.0 / idx
                break
        mrrs.append(rr)
        # CTR by position (relevance as click)
        for idx, item_id in enumerate(items[:10], start=1):
            ctr_impr[idx] = ctr_impr.get(idx, 0) + 1
            if _relevance(item_id, liked, visited, dwell) > 0:
                ctr_clicks[idx] = ctr_clicks.get(idx, 0) + 1
    ctr_by_position = {str(k): round(ctr_clicks.get(k, 0) / ctr_impr[k], 4) for k in ctr_impr}
    return {
        "ndcg_at_10": round(sum(ndcgs) / len(ndcgs), 4),
        "mrr": round(sum(mrrs) / len(mrrs), 4),
        "ctr_by_position": ctr_by_position,
        "session_success_rate": round(success / len(_SESSION_LOG), 4),
    }
