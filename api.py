"""FastAPI service.  Run:  uvicorn api:app --reload"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from retainai import agents, tools  # noqa: E402

app = FastAPI(title="RetainAI", description="Multi-agent customer retention assistant")


@lru_cache(maxsize=1)
def graph():
    return agents.build_graph()


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "question is empty")
    return agents.ask(req.question, graph=graph())


@app.get("/customers/{customer_id}/risk")
def customer_risk(customer_id: str):
    out = tools.get_churn_risk.invoke({"customer_id": customer_id})
    if "error" in out:
        raise HTTPException(404, out["error"])
    return out


@app.get("/customers/at-risk")
def at_risk(limit: int = 10, min_monthly_charges: float = 0.0):
    return tools.list_high_risk_customers.invoke({"limit": limit, "min_monthly_charges": min_monthly_charges})
