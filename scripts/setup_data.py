"""Download the Telco churn dataset, load it into SQLite, train the model and build the RAG index.

Usage:  python scripts/setup_data.py            # everything
        python scripts/setup_data.py --skip-index
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retainai import config, db, model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-index", action="store_true", help="don't build the Chroma index")
    args = ap.parse_args()

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not config.RAW_CSV.exists():
        print(f"Downloading dataset -> {config.RAW_CSV}")
        urllib.request.urlretrieve(config.DATASET_URL, config.RAW_CSV)

    n = db.load_csv_to_sqlite()
    print(f"Loaded {n:,} customers into {config.DB_PATH}")

    r = model.train()
    print(f"Churn model: hold-out ROC-AUC {r.test_roc_auc}, PR-AUC {r.test_pr_auc}, "
          f"5-fold out-of-fold ROC-AUC {r.oof_roc_auc} (churn rate {r.churn_rate:.1%})")

    if not args.skip_index:
        from retainai import rag
        chunks = rag.build_index()
        print(f"Indexed {chunks} policy chunks into {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
