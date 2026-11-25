"""Shared fixtures: a tiny copy of the data pipeline in a temp dir, and a scripted fake LLM."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def built_data(tmp_path_factory):
    """Run the real pipeline (load, train, index) once into a temp data dir."""
    data_dir = tmp_path_factory.mktemp("data")
    src_csv = ROOT / "data" / "telco_churn.csv"
    if not src_csv.exists():
        pytest.skip("data/telco_churn.csv missing: run scripts/setup_data.py first")
    shutil.copy(src_csv, data_dir / "telco_churn.csv")
    os.environ["RETAINAI_EMBEDDINGS"] = "hashing"  # offline embeddings in tests

    from retainai import config
    config.DATA_DIR = data_dir
    config.RAW_CSV = data_dir / "telco_churn.csv"
    config.DB_PATH = data_dir / "retainai.db"
    config.MODEL_PATH = data_dir / "churn_model.json"
    config.MODEL_META_PATH = data_dir / "churn_model_meta.json"
    config.CHROMA_DIR = data_dir / "chroma"

    from retainai import db, model, rag
    db.load_csv_to_sqlite(config.RAW_CSV, config.DB_PATH)
    result = model.train(config.DB_PATH, config.MODEL_PATH, config.MODEL_META_PATH)
    rag.build_index(config.POLICY_DIR, config.CHROMA_DIR)
    return result


class ScriptedLLM:
    """Stands in for a chat model. Returns pre-written responses in order, so the graph's
    control flow can be tested without an API key."""

    def __init__(self, routes, worker_msgs, final="Final answer."):
        self.routes = list(routes)        # Route objects for the supervisor
        self.worker_msgs = list(worker_msgs)  # AIMessages for workers (tool calls, then text)
        self.final = final

    def bind_tools(self, tools):
        return _Proxy(lambda msgs: self.worker_msgs.pop(0))

    def with_structured_output(self, schema):
        return _Proxy(lambda msgs: self.routes.pop(0))

    def invoke(self, msgs):
        from langchain_core.messages import AIMessage
        return AIMessage(content=self.final)


class _Proxy:
    def __init__(self, fn):
        self.fn = fn

    def invoke(self, msgs):
        return self.fn(msgs)
