"""Paths and settings shared across the project."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("RETAINAI_DATA_DIR", ROOT / "data"))
RAW_CSV = DATA_DIR / "telco_churn.csv"
DB_PATH = DATA_DIR / "retainai.db"
MODEL_PATH = DATA_DIR / "churn_model.json"
MODEL_META_PATH = DATA_DIR / "churn_model_meta.json"
CHROMA_DIR = DATA_DIR / "chroma"
POLICY_DIR = ROOT / "docs" / "policies"

DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

LLM_MODEL = os.getenv("RETAINAI_MODEL", "gemini-3.5-flash")
MAX_SUPERVISOR_STEPS = int(os.getenv("RETAINAI_MAX_STEPS", "6"))
