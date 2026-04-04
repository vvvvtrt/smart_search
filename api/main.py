from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import os
import sqlite3
from core.search_agent import run_agent, fetch_item_sql, DB_PATH, ensure_db_ready
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
    search_context: Optional[Dict[str, Any]] = None
    page: int = 1
    limit: int = 5
    session_feedback: Optional[Dict[str, Any]] = None

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
        conn.close()
    except Exception:
        build_db()


@app.post("/search")
def search(payload: SearchPayload):
    return run_agent(payload.model_dump())


@app.get("/companies")
def companies(kind: str = "customer"):
    payload = {"user_query": "", "user_profile": {}, "mode": "companies_only"}
    result = run_agent(payload)
    if kind == "supplier":
        return {"companies": result.get("supplier_companies", [])}
    return {"companies": result.get("customer_companies", [])}


@app.get("/item/{item_id}")
def item(item_id: str):
    ensure_db_ready()
    conn = sqlite3.connect(DB_PATH)
    data = fetch_item_sql(conn, item_id)
    conn.close()
    return data or {"error": "not_found"}


app.mount("/", StaticFiles(directory="web", html=True), name="web")
