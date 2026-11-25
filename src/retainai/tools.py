"""Tools the agents can call. Each returns JSON-serialisable data."""
from __future__ import annotations

import sqlite3
from functools import lru_cache

import pandas as pd
from langchain_core.tools import tool

from . import config, db, model
from .rag import PolicyRetriever


def _customer_row(customer_id: str) -> pd.Series | None:
    with sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True) as con:
        df = pd.read_sql("SELECT * FROM customers WHERE customer_id = ?", con, params=[customer_id.strip()])
    return None if df.empty else df.iloc[0]


def _score(customer_id: str) -> dict | None:
    with sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True) as con:
        row = con.execute("SELECT churn_probability, risk_band FROM churn_scores WHERE customer_id = ?",
                          [customer_id.strip()]).fetchone()
    return None if row is None else {"churn_probability": row[0], "risk_band": row[1]}


@lru_cache(maxsize=1)
def _retriever() -> PolicyRetriever:
    return PolicyRetriever()


# --- SQL analytics agent ----------------------------------------------------

@tool
def get_database_schema() -> str:
    """Describe the tables and columns available for SQL queries."""
    return db.schema_description()


@tool
def run_sql_query(query: str) -> dict:
    """Run ONE read-only SQLite SELECT query against the customers and churn_scores tables.
    Returns up to 50 rows. Use aggregates (COUNT, AVG, SUM, GROUP BY) for summary questions."""
    try:
        return db.run_select(query)
    except Exception as e:  # return the error so the agent can fix its query
        return {"error": str(e)}


# --- Churn scoring agent ----------------------------------------------------

@tool
def get_churn_risk(customer_id: str) -> dict:
    """Churn probability, risk band and the top SHAP drivers for one customer."""
    row = _customer_row(customer_id)
    if row is None:
        return {"error": f"No customer with id {customer_id!r}"}
    score = _score(customer_id) or {}
    return {
        "customer_id": row["customer_id"],
        **score,
        "monthly_charges": float(row["monthly_charges"]),
        "revenue_at_risk_per_year": round(score.get("churn_probability", 0) * float(row["monthly_charges"]) * 12, 2),
        "top_drivers": model.explain_customer(row),
    }


@tool
def list_high_risk_customers(limit: int = 10, min_monthly_charges: float = 0.0,
                             contract: str | None = None, internet_service: str | None = None) -> dict:
    """Customers who have not churned yet, ranked by revenue at risk
    (churn probability x monthly charges x 12). Optional filters: minimum monthly charges,
    contract ('Month-to-month', 'One year', 'Two year'), internet_service ('DSL', 'Fiber optic', 'No')."""
    sql = """
        SELECT c.customer_id, s.churn_probability, s.risk_band, c.monthly_charges, c.tenure_months,
               c.contract, c.internet_service, c.payment_method,
               ROUND(s.churn_probability * c.monthly_charges * 12, 2) AS revenue_at_risk_per_year
        FROM customers c JOIN churn_scores s USING (customer_id)
        WHERE c.churned = 0 AND c.monthly_charges >= ?
    """
    params: list = [min_monthly_charges]
    if contract:
        sql += " AND c.contract = ?"
        params.append(contract)
    if internet_service:
        sql += " AND c.internet_service = ?"
        params.append(internet_service)
    sql += " ORDER BY revenue_at_risk_per_year DESC LIMIT ?"
    params.append(max(1, min(int(limit), 50)))
    with sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True) as con:
        df = pd.read_sql(sql, con, params=params)
    return {"customers": df.to_dict(orient="records")}


# --- Offer recommendation agent ---------------------------------------------

@tool
def get_customer_profile(customer_id: str) -> dict:
    """All attributes of one customer (contract, tenure, services, payment method, charges)
    plus their churn risk. Use it to check offer eligibility."""
    row = _customer_row(customer_id)
    if row is None:
        return {"error": f"No customer with id {customer_id!r}"}
    profile = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
    return {**profile, **(_score(customer_id) or {})}


@tool
def search_retention_policies(query: str) -> list[dict]:
    """Search the retention offer catalogue, discount approval policy, plan reference and
    contact guidelines. Returns the 3 most relevant sections."""
    return _retriever().search(query, k=3)


ANALYTICS_TOOLS = [get_database_schema, run_sql_query]
CHURN_TOOLS = [get_churn_risk, list_high_risk_customers]
OFFER_TOOLS = [get_customer_profile, search_retention_policies]
