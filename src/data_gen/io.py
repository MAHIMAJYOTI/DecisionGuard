"""Write the raw extracts and a manifest that describes the run.

CSV and Parquet carry the same rows. CSV is what a lot of source systems
still drop in a folder. Parquet keeps types (including blanks in integer
columns) and is what the warehouse load will prefer.

The manifest is produced by reading the frames back. It is a receipt for
the run, not a set of numbers typed in by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_gen.config import project_root
from src.data_gen.dataset import GeneratedDataset
from src.data_gen.quality import QualityIssue

TABLE_KEYS = {
    "customers": "customer_id",
    "credit_bureau": "customer_id",
    "loan_applications": "application_id",
    "transactions": "txn_id",
    "loan_performance": "application_id",
}


def default_output_dir() -> Path:
    return project_root() / "data" / "raw"


def write_raw_files(dataset: GeneratedDataset, output_dir: Path | None = None) -> Path:
    """Write every table twice (CSV and Parquet) and return the manifest path."""

    output_dir = output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in dataset.tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(dataset), indent=2), encoding="utf-8")
    return manifest_path


def _manifest(dataset: GeneratedDataset) -> dict[str, object]:
    return {
        "seed": dataset.seed,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "notes": list(dataset.notes),
        "clean_summary": dataset.clean_summary,
        "injected_issues": [_issue_dict(issue) for issue in dataset.issues],
        "raw_files": _profile(dataset.tables),
    }


def _issue_dict(issue: QualityIssue) -> dict[str, object]:
    return {
        "table": issue.table,
        "issue": issue.issue,
        "rows_affected": issue.rows_affected,
        "detail": issue.detail,
    }


def _profile(tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    """What is actually in the files after defects, counted from the frames."""

    profile: dict[str, object] = {}
    for name, frame in tables.items():
        key = TABLE_KEYS[name]
        nulls = {
            column: int(frame[column].isna().sum())
            for column in frame.columns
            if int(frame[column].isna().sum()) > 0
        }
        profile[name] = {
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "extra_duplicate_key_rows": int(frame[key].duplicated().sum()),
            "null_counts": nulls,
        }
    transactions = tables["transactions"]
    applications = tables["loan_applications"]
    profile["transactions"]["negative_amounts"] = int((transactions["amount"].dropna() < 0).sum())
    profile["loan_applications"]["negative_requested_amounts"] = int(
        (applications["requested_amount"].dropna() < 0).sum()
    )
    return profile
