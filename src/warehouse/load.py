"""Land every raw CSV in DuckDB, all columns as text.

In production this step is a BigQuery load job (or an external table over
files in cloud storage) with every column declared as STRING. DuckDB's
`read_csv(..., all_varchar=true)` is the same idea on a laptop: nothing is
parsed until staging asks for a type with TRY_CAST.

Two technical columns are added, the way an ingestion job stamps a file:
`_loaded_at` and `_source_file`. `_src_row` is the position in the file so
"keep the first copy of a duplicated id" has a stable meaning.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from src.data_gen.config import (
    APPLICATION_COLUMNS,
    BUREAU_COLUMNS,
    CUSTOMER_COLUMNS,
    PERFORMANCE_COLUMNS,
    TRANSACTION_COLUMNS,
    project_root,
)

# Business columns only. The loader adds the technical columns itself.
RAW_TABLES: dict[str, tuple[str, ...]] = {
    "customers": CUSTOMER_COLUMNS,
    "credit_bureau": BUREAU_COLUMNS,
    "loan_applications": APPLICATION_COLUMNS,
    "transactions": TRANSACTION_COLUMNS,
    "loan_performance": PERFORMANCE_COLUMNS,
}


def default_raw_dir() -> Path:
    return project_root() / "data" / "raw"


def default_db_path() -> Path:
    return project_root() / "data" / "warehouse.duckdb"


def load_raw(raw_dir: Path | None = None, db_path: Path | None = None) -> Path:
    """Create (or replace) the raw schema and return the database path."""

    raw_dir = raw_dir or default_raw_dir()
    db_path = db_path or default_db_path()
    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest at {manifest_path}. Run `python -m src.data_gen` first."
        )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(db_path))
    try:
        connection.execute("CREATE SCHEMA IF NOT EXISTS raw")
        for table_name, columns in RAW_TABLES.items():
            csv_path = raw_dir / f"{table_name}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing raw file: {csv_path}")
            _load_table(connection, table_name, columns, csv_path)
        _load_expected_issues(connection, manifest_path)
    finally:
        connection.close()
    return db_path


def _load_table(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    columns: tuple[str, ...],
    csv_path: Path,
) -> None:
    """Read one CSV. Every business column is forced to VARCHAR."""

    # CAST is explicit even though all_varchar already returns text, so a
    # reader can see the contract without knowing the DuckDB flag.
    cast_list = ",\n                ".join(
        f"CAST({column} AS VARCHAR) AS {column}" for column in columns
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TABLE raw.{table_name} AS
        SELECT
                {cast_list},
                CAST(row_number() OVER () AS BIGINT) AS _src_row,
                CURRENT_TIMESTAMP AS _loaded_at,
                ? AS _source_file
        FROM read_csv(?, header = true, all_varchar = true)
        """,
        [csv_path.name, csv_path.as_posix()],
    )
    row_count = connection.execute(f"SELECT count(*) FROM raw.{table_name}").fetchone()
    loaded = int(row_count[0]) if row_count else 0
    print(f"  raw.{table_name:<22} {loaded:>8,} rows  ({csv_path.name})")


def _load_expected_issues(connection: duckdb.DuckDBPyConnection, manifest_path: Path) -> None:
    """Store the generator's defect list so dbt can check it still matches.

    The numbers come from manifest.json, which the generator wrote while it
    injected the defects. The reconciliation test fails if staging disagrees.
    """

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    issues = manifest["injected_issues"]
    connection.execute(
        """
        CREATE OR REPLACE TABLE raw.expected_issues (
            table_name VARCHAR,
            issue VARCHAR,
            rows_affected BIGINT
        )
        """
    )
    connection.executemany(
        "INSERT INTO raw.expected_issues VALUES (?, ?, ?)",
        [(item["table"], item["issue"], int(item["rows_affected"])) for item in issues],
    )
    print(f"  raw.expected_issues        {len(issues):>8,} defect types")
