"""Print the Phase 2 reconciliation and fraud-feature comparison.

The numbers are read from the warehouse after `dbt build`. Nothing here is
typed in by hand.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from src.warehouse.load import default_db_path

FEATURE_COLUMNS = (
    "txn_count_1h",
    "txn_amount_1h",
    "txn_count_24h",
    "txn_amount_24h",
    "avg_amount_30d",
    "amount_vs_avg_30d",
    "is_new_device",
    "is_foreign_country",
    "is_night",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print Phase 2 warehouse summary.")
    parser.add_argument("--db-path", type=Path, default=default_db_path())
    args = parser.parse_args()
    connection = duckdb.connect(str(args.db_path), read_only=True)
    try:
        _print_reconciliation(connection)
        _print_fraud_features(connection)
    finally:
        connection.close()


def _print_reconciliation(connection: duckdb.DuckDBPyConnection) -> None:
    rows = connection.execute(
        """
        WITH tables AS (
            SELECT * FROM (VALUES
                ('customers'),
                ('credit_bureau'),
                ('loan_applications'),
                ('transactions'),
                ('loan_performance')
            ) AS t(table_name)
        ),
        raw_counts AS (
            SELECT 'customers' AS table_name, count(*) AS raw_rows FROM raw.customers
            UNION ALL SELECT 'credit_bureau', count(*) FROM raw.credit_bureau
            UNION ALL SELECT 'loan_applications', count(*) FROM raw.loan_applications
            UNION ALL SELECT 'transactions', count(*) FROM raw.transactions
            UNION ALL SELECT 'loan_performance', count(*) FROM raw.loan_performance
        ),
        valid_counts AS (
            SELECT 'customers' AS table_name, count(*) AS staging_rows FROM staging.stg_customers
            UNION ALL SELECT 'credit_bureau', count(*) FROM staging.stg_credit_bureau
            UNION ALL SELECT 'loan_applications', count(*) FROM staging.stg_loan_applications
            UNION ALL SELECT 'transactions', count(*) FROM staging.stg_transactions
            UNION ALL SELECT 'loan_performance', count(*) FROM staging.stg_loan_performance
        ),
        rejected AS (
            SELECT table_name, count(DISTINCT _src_row) AS rejected_rows
            FROM staging.stg_rejected_records
            WHERE _src_row IS NOT NULL
            GROUP BY table_name
        )
        SELECT
            t.table_name,
            r.raw_rows,
            v.staging_rows,
            coalesce(j.rejected_rows, 0) AS rejected_rows
        FROM tables t
        JOIN raw_counts r USING (table_name)
        JOIN valid_counts v USING (table_name)
        LEFT JOIN rejected j USING (table_name)
        ORDER BY t.table_name
        """
    ).fetchall()

    print("Raw vs staging vs rejected (rejected = distinct raw rows with at least one reason)")
    print(f"{'table':<22} {'raw':>10} {'staging':>10} {'rejected':>10} {'sum':>10}")
    for table_name, raw_rows, staging_rows, rejected_rows in rows:
        print(
            f"{table_name:<22} {raw_rows:>10,} {staging_rows:>10,} "
            f"{rejected_rows:>10,} {staging_rows + rejected_rows:>10,}"
        )
    print()
    print("Rejected reason rows (one raw row can have more than one reason)")
    reason_rows = connection.execute(
        """
        SELECT table_name, reject_reason, rejected_rows
        FROM marts.mart_data_quality
        ORDER BY table_name, rejected_rows DESC
        """
    ).fetchall()
    for table_name, reason, count in reason_rows:
        print(f"  {table_name:<22} {reason:<32} {count:>6,}")


def _print_fraud_features(connection: duckdb.DuckDBPyConnection) -> None:
    averages = ", ".join(f"avg({column}) AS {column}" for column in FEATURE_COLUMNS)
    rows = connection.execute(
        f"""
        SELECT is_fraud, count(*) AS txns, {averages}
        FROM marts.mart_fraud_transactions
        GROUP BY is_fraud
        ORDER BY is_fraud
        """
    ).fetchall()
    print()
    print("Average fraud features by label (from mart_fraud_transactions)")
    # Print a row per feature so the two groups line up even if a group is missing.
    by_label = {int(row[0]): row for row in rows}
    labels = [label for label in (0, 1) if label in by_label]
    print(f"{'feature':<22}" + "".join(f"{('fraud' if label else 'non-fraud'):>18}" for label in labels))
    print(f"{'rows':<22}" + "".join(f"{by_label[label][1]:>18,}" for label in labels))
    for feature_index, feature in enumerate(FEATURE_COLUMNS, start=2):
        cells = []
        for label in labels:
            value = by_label[label][feature_index]
            cells.append(f"{float(value):>18.4f}" if value is not None else f"{'null':>18}")
        print(f"{feature:<22}" + "".join(cells))


if __name__ == "__main__":
    main()
