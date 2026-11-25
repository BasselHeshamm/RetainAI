"""Load the Telco churn CSV into SQLite and run read-only queries against it."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

from . import config

CUSTOMER_COLUMNS = {
    "customer_id": "Unique customer id, e.g. '7590-VHVEG'",
    "gender": "'Male' or 'Female'",
    "senior_citizen": "1 if the customer is 65 or older, else 0",
    "partner": "'Yes' / 'No'",
    "dependents": "'Yes' / 'No'",
    "tenure_months": "Months the customer has been with the company",
    "phone_service": "'Yes' / 'No'",
    "multiple_lines": "'Yes' / 'No' / 'No phone service'",
    "internet_service": "'DSL' / 'Fiber optic' / 'No'",
    "online_security": "'Yes' / 'No' / 'No internet service'",
    "online_backup": "'Yes' / 'No' / 'No internet service'",
    "device_protection": "'Yes' / 'No' / 'No internet service'",
    "tech_support": "'Yes' / 'No' / 'No internet service'",
    "streaming_tv": "'Yes' / 'No' / 'No internet service'",
    "streaming_movies": "'Yes' / 'No' / 'No internet service'",
    "contract": "'Month-to-month' / 'One year' / 'Two year'",
    "paperless_billing": "'Yes' / 'No'",
    "payment_method": "'Electronic check' / 'Mailed check' / 'Bank transfer (automatic)' / 'Credit card (automatic)'",
    "monthly_charges": "Current monthly bill in USD",
    "total_charges": "Total billed to date in USD (0 for brand-new customers)",
    "churned": "1 if the customer left in the last month, else 0",
}

SCORE_COLUMNS = {
    "customer_id": "Joins to customers.customer_id",
    "churn_probability": "Model's out-of-fold probability (0-1) that the customer churns",
    "risk_band": "'High' (>=0.6), 'Medium' (0.3-0.6) or 'Low' (<0.3)",
}

_RENAME = {
    "customerID": "customer_id",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure_months",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churned",
}


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to snake_case and fix types (TotalCharges has blanks for new customers)."""
    df = df.rename(columns=_RENAME)
    df["total_charges"] = pd.to_numeric(df["total_charges"].astype(str).str.strip(), errors="coerce").fillna(0.0)
    df["churned"] = (df["churned"] == "Yes").astype(int)
    return df[list(CUSTOMER_COLUMNS)]


def load_csv_to_sqlite(csv_path: Path | None = None, db_path: Path | None = None) -> int:
    csv_path, db_path = csv_path or config.RAW_CSV, db_path or config.DB_PATH
    df = clean_raw(pd.read_csv(csv_path))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        df.to_sql("customers", con, if_exists="replace", index=False)
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_customers_id ON customers(customer_id)")
    return len(df)


def read_customers(db_path: Path | None = None) -> pd.DataFrame:
    db_path = db_path or config.DB_PATH
    with sqlite3.connect(db_path) as con:
        return pd.read_sql("SELECT * FROM customers", con)


def write_scores(scores: pd.DataFrame, db_path: Path | None = None) -> None:
    db_path = db_path or config.DB_PATH
    with sqlite3.connect(db_path) as con:
        scores.to_sql("churn_scores", con, if_exists="replace", index=False)
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_scores_id ON churn_scores(customer_id)")


# --- Read-only SQL for the analytics agent ---------------------------------

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex|truncate)\b",
    re.IGNORECASE,
)


class UnsafeQueryError(ValueError):
    pass


def validate_select(sql: str) -> str:
    """Allow a single SELECT (or WITH ... SELECT) statement only."""
    stripped = sql.strip().rstrip(";").strip()
    if ";" in stripped:
        raise UnsafeQueryError("Only one statement is allowed.")
    if not re.match(r"^(select|with)\b", stripped, re.IGNORECASE):
        raise UnsafeQueryError("Only SELECT queries are allowed.")
    if _FORBIDDEN.search(stripped):
        raise UnsafeQueryError("Query contains a forbidden keyword.")
    return stripped


def run_select(sql: str, db_path: Path | None = None, max_rows: int = 50) -> dict:
    query = validate_select(sql)
    db_path = db_path or config.DB_PATH
    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as con:
        cur = con.execute(query)
        cols = [c[0] for c in cur.description]
        rows = cur.fetchmany(max_rows + 1)
    truncated = len(rows) > max_rows
    return {"columns": cols, "rows": [list(r) for r in rows[:max_rows]], "truncated": truncated}


def schema_description() -> str:
    lines = ["Table customers (one row per customer):"]
    lines += [f"  - {c}: {d}" for c, d in CUSTOMER_COLUMNS.items()]
    lines.append("Table churn_scores (one row per customer, from the churn model):")
    lines += [f"  - {c}: {d}" for c, d in SCORE_COLUMNS.items()]
    return "\n".join(lines)
