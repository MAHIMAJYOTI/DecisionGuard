"""Phase 2: load a small book and run dbt build, including the reconciliation tests."""

from __future__ import annotations

import os
from pathlib import Path

from dbt.cli.main import dbtRunner

from src.data_gen.config import DataGenConfig, project_root
from src.data_gen.dataset import generate_dataset
from src.data_gen.io import write_raw_files
from src.warehouse.load import load_raw


def test_dbt_build_on_small_book(tmp_path: Path) -> None:
    """A smaller extract still has every kind of defect, so the same tests apply."""

    raw_dir = tmp_path / "raw"
    db_path = tmp_path / "warehouse.duckdb"
    dataset = generate_dataset(
        DataGenConfig(
            seed=42,
            n_customers=800,
            n_applications=800,
            n_transactions=5_000,
        )
    )
    write_raw_files(dataset, raw_dir)
    load_raw(raw_dir, db_path)

    previous = os.environ.get("DECISIONGUARD_DUCKDB")
    os.environ["DECISIONGUARD_DUCKDB"] = str(db_path)
    try:
        result = dbtRunner().invoke(
            [
                "build",
                "--project-dir",
                str(project_root() / "dbt"),
                "--profiles-dir",
                str(project_root() / "dbt"),
            ]
        )
    finally:
        if previous is None:
            os.environ.pop("DECISIONGUARD_DUCKDB", None)
        else:
            os.environ["DECISIONGUARD_DUCKDB"] = previous

    assert result.success, result.exception
