"""Load data/raw into data/warehouse.duckdb.

    python -m src.warehouse
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.warehouse.load import default_db_path, default_raw_dir, load_raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Load DecisionGuard raw CSVs into DuckDB.")
    parser.add_argument("--raw-dir", type=Path, default=default_raw_dir())
    parser.add_argument("--db-path", type=Path, default=default_db_path())
    args = parser.parse_args()

    print(f"Loading {args.raw_dir} into {args.db_path}")
    load_raw(args.raw_dir, args.db_path)
    print()
    print("Next, build the dbt models (PowerShell):")
    print(f'  $env:DECISIONGUARD_DUCKDB = "{args.db_path.resolve()}"')
    print("  dbt build --project-dir dbt --profiles-dir dbt")


if __name__ == "__main__":
    main()
