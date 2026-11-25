"""XGBoost churn model: feature engineering, training, scoring and SHAP explanations."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import config, db

CATEGORICAL = [
    "gender", "partner", "dependents", "phone_service", "multiple_lines", "internet_service",
    "online_security", "online_backup", "device_protection", "tech_support", "streaming_tv",
    "streaming_movies", "contract", "paperless_billing", "payment_method",
]
NUMERIC = ["senior_citizen", "tenure_months", "monthly_charges", "total_charges"]

PARAMS = dict(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_lambda=1.0,
    eval_metric="logloss",
    random_state=42,
)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw customer rows -> model features (one-hot categoricals + a few ratios)."""
    X = df[NUMERIC].astype(float).copy()
    X["avg_monthly_spend"] = np.where(df["tenure_months"] > 0, df["total_charges"] / df["tenure_months"].clip(lower=1), df["monthly_charges"])
    X["charge_increase"] = df["monthly_charges"] - X["avg_monthly_spend"]
    addons = ["online_security", "online_backup", "device_protection", "tech_support", "streaming_tv", "streaming_movies"]
    X["n_addons"] = (df[addons] == "Yes").sum(axis=1)
    X["is_new_customer"] = (df["tenure_months"] <= 6).astype(int)
    dummies = pd.get_dummies(df[CATEGORICAL], prefix=CATEGORICAL, dtype=int)
    return pd.concat([X, dummies], axis=1)


def risk_band(p: float) -> str:
    return "High" if p >= 0.6 else "Medium" if p >= 0.3 else "Low"


@dataclass
class TrainResult:
    test_roc_auc: float
    test_pr_auc: float
    oof_roc_auc: float
    n_customers: int
    churn_rate: float


def train(db_path: Path | None = None, model_path: Path | None = None,
          meta_path: Path | None = None) -> TrainResult:
    db_path = db_path or config.DB_PATH
    model_path = model_path or config.MODEL_PATH
    meta_path = meta_path or config.MODEL_META_PATH
    df = db.read_customers(db_path)
    X = engineer_features(df)
    y = df["churned"].values

    # 1) Honest hold-out estimate of performance.
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = xgb.XGBClassifier(**PARAMS)
    clf.fit(X_tr, y_tr)
    p_te = clf.predict_proba(X_te)[:, 1]
    test_auc = roc_auc_score(y_te, p_te)
    test_pr = average_precision_score(y_te, p_te)

    # 2) Out-of-fold scores for every customer, so stored scores never come
    #    from a model that saw that customer during training (no leakage).
    oof = np.zeros(len(df))
    for tr_idx, va_idx in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        m = xgb.XGBClassifier(**PARAMS)
        m.fit(X.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = m.predict_proba(X.iloc[va_idx])[:, 1]
    oof_auc = roc_auc_score(y, oof)
    scores = pd.DataFrame({"customer_id": df["customer_id"], "churn_probability": oof.round(4)})
    scores["risk_band"] = scores["churn_probability"].map(risk_band)
    db.write_scores(scores, db_path)

    # 3) Final model on all data, used to explain individual customers.
    final = xgb.XGBClassifier(**PARAMS)
    final.fit(X, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    final.save_model(model_path)
    result = TrainResult(round(test_auc, 4), round(test_pr, 4), round(oof_auc, 4), len(df), round(float(y.mean()), 4))
    meta_path.write_text(json.dumps({"features": list(X.columns), **result.__dict__}, indent=2))
    _load_model.cache_clear()
    return result


@lru_cache(maxsize=4)
def _load_model(model_path: str, meta_path: str):
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    meta = json.loads(Path(meta_path).read_text())
    return model, meta, shap.TreeExplainer(model)


def _align(X: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return X.reindex(columns=features, fill_value=0)


def explain_customer(row: pd.Series, top_k: int = 5, model_path: Path | None = None,
                     meta_path: Path | None = None) -> list[dict]:
    """Top SHAP drivers for one customer, merged back to readable feature names."""
    model_path = model_path or config.MODEL_PATH
    meta_path = meta_path or config.MODEL_META_PATH
    model, meta, explainer = _load_model(str(model_path), str(meta_path))
    X = _align(engineer_features(row.to_frame().T.infer_objects()), meta["features"])
    values = explainer.shap_values(X)[0]
    contrib: dict[str, float] = {}
    for feat, v in zip(meta["features"], values):
        base = next((c for c in CATEGORICAL if feat.startswith(c + "_")), feat)
        contrib[base] = contrib.get(base, 0.0) + float(v)
    ranked = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    out = []
    for feat, v in ranked:
        value = row[feat] if feat in row.index else (round(float(X.iloc[0][feat]), 2) if feat in X.columns else None)
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        out.append({"feature": feat, "value": value, "shap": round(v, 3),
                    "effect": "raises churn risk" if v > 0 else "lowers churn risk"})
    return out
