"""Export a small demo snapshot from the warehouse.

The Streamlit app reads only app/data, app/models and app/strategies.
It does not open DuckDB. Numbers in the snapshot are queried or scored
here, then written out. Nothing is typed in by hand.

    python -m src.demo.export
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from src.data_gen.config import project_root
from src.decision_engine.engine import decide_frame
from src.decision_engine.strategy import load_strategy
from src.models.store import load_metadata, load_model
from src.validation.report import recommend
from src.warehouse.load import default_db_path

CREDIT_SAMPLE_ROWS = 5_000
FRAUD_SAMPLE_ROWS = 20_000
DATA_BUDGET_BYTES = 25 * 1024 * 1024
FILE_BUDGET_BYTES = 50 * 1024 * 1024

STRATEGY_FILES = (
    "credit_strategy_v1.yaml",
    "credit_strategy_v2.yaml",
    "fraud_strategy_v1.yaml",
    "fraud_strategy_v2.yaml",
)


def export_demo(db_path: Path | None = None) -> Path:
    root = project_root()
    path = db_path or default_db_path()
    if not path.exists():
        raise FileNotFoundError(f"No warehouse at {path}. Load it before exporting the demo.")
    data_dir = root / "app" / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)
    connection = duckdb.connect(str(path), read_only=True)
    try:
        _write_portfolio(connection, data_dir)
        _write_quality(connection, data_dir)
        _write_fraud_features(connection, data_dir)
        _write_samples(connection, data_dir)
    finally:
        connection.close()
    decisions = _write_decisions(path, data_dir)
    golden = _golden(root)
    _write_models_and_strategy(root, data_dir, decisions, golden)
    _copy_runtime(root)
    _check_sizes(root)
    print(f"Demo snapshot ready in {root / 'app'}")
    return data_dir


def _write_portfolio(connection: duckdb.DuckDBPyConnection, data_dir: Path) -> None:
    summary = connection.execute("SELECT * FROM marts.mart_portfolio_summary").fetchdf()
    summary.to_parquet(data_dir / "portfolio_summary.parquet", index=False)
    kpis = connection.execute(
        """
        SELECT
            (SELECT count(*) FROM staging.stg_customers) AS customers,
            (SELECT count(*) FROM marts.mart_credit_applications) AS applications,
            (SELECT count(*) FROM marts.mart_credit_applications WHERE was_booked = 1) AS booked,
            (SELECT count(*) FROM marts.mart_credit_applications WHERE outcome_status = 'good') AS good_loans,
            (SELECT count(*) FROM marts.mart_credit_applications WHERE outcome_status = 'bad') AS bad_loans,
            (SELECT count(*) FROM marts.mart_credit_applications WHERE outcome_status = 'not_yet_known') AS not_yet_known,
            (SELECT count(*) FROM marts.mart_credit_applications WHERE outcome_status = 'not_booked') AS not_booked,
            (SELECT cast(round(sum(disbursed_amount)) AS BIGINT) FROM marts.mart_credit_applications) AS disbursed_amount,
            (SELECT count(*) FROM marts.mart_fraud_transactions) AS transactions,
            (SELECT sum(is_fraud) FROM marts.mart_fraud_transactions) AS frauds
        """
    ).fetchdf()
    row = kpis.iloc[0]
    payload = {column: _number(row[column]) for column in kpis.columns}
    payload["fraud_rate"] = payload["frauds"] / payload["transactions"] if payload["transactions"] else None
    _dump(data_dir / "portfolio_kpis.json", payload)
    print(
        f"Portfolio: {payload['applications']:,} applications, "
        f"{payload['transactions']:,} transactions, fraud rate {payload['fraud_rate']:.4f}"
    )


def _write_quality(connection: duckdb.DuckDBPyConnection, data_dir: Path) -> None:
    tables = connection.execute(
        """
        WITH raw_counts AS (
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
            r.table_name,
            r.raw_rows,
            v.staging_rows,
            coalesce(j.rejected_rows, 0) AS rejected_rows
        FROM raw_counts r
        JOIN valid_counts v USING (table_name)
        LEFT JOIN rejected j USING (table_name)
        ORDER BY r.table_name
        """
    ).fetchdf()
    reasons = connection.execute(
        """
        SELECT table_name, reject_reason, rejected_rows
        FROM marts.mart_data_quality
        ORDER BY table_name, rejected_rows DESC
        """
    ).fetchdf()
    _dump(
        data_dir / "data_quality.json",
        {
            "tables": tables.to_dict(orient="records"),
            "reasons": reasons.to_dict(orient="records"),
            "rejected_meaning": (
                "rejected_rows counts distinct raw rows. "
                "A row with two defects is one rejected row and two reason rows."
            ),
        },
    )
    print("Data quality tables:", ", ".join(tables["table_name"]))


def _write_fraud_features(connection: duckdb.DuckDBPyConnection, data_dir: Path) -> None:
    frame = connection.execute(
        """
        SELECT
            is_fraud,
            count(*) AS transactions,
            avg(txn_count_1h) AS txn_count_1h,
            avg(txn_amount_1h) AS txn_amount_1h,
            avg(txn_count_24h) AS txn_count_24h,
            avg(txn_amount_24h) AS txn_amount_24h,
            avg(avg_amount_30d) AS avg_amount_30d,
            avg(amount_vs_avg_30d) AS amount_vs_avg_30d,
            avg(is_new_device) AS is_new_device,
            avg(is_foreign_country) AS is_foreign_country,
            avg(is_night) AS is_night
        FROM marts.mart_fraud_transactions
        GROUP BY is_fraud
        ORDER BY is_fraud
        """
    ).fetchdf()
    _dump(data_dir / "fraud_features.json", {"by_label": frame.to_dict(orient="records")})


def _write_samples(connection: duckdb.DuckDBPyConnection, data_dir: Path) -> None:
    credit = connection.execute(
        """
        WITH sized AS (
            SELECT outcome_status, count(*) AS n
            FROM marts.mart_credit_applications
            GROUP BY outcome_status
        ),
        ranked AS (
            SELECT
                applications.*,
                row_number() OVER (
                    PARTITION BY applications.outcome_status
                    ORDER BY hash(applications.application_id)
                ) AS sample_rank,
                sized.n
            FROM marts.mart_credit_applications AS applications
            JOIN sized USING (outcome_status)
        )
        SELECT * EXCLUDE (sample_rank, n)
        FROM ranked
        WHERE sample_rank <= greatest(1, cast(round(? * n / (SELECT sum(n) FROM sized)) AS INTEGER))
        """,
        [CREDIT_SAMPLE_ROWS],
    ).fetchdf()
    fraud_counts = connection.execute(
        "SELECT is_fraud, count(*) AS n FROM marts.mart_fraud_transactions GROUP BY is_fraud"
    ).fetchdf()
    total = int(fraud_counts["n"].sum())
    quotas = {}
    for _, row in fraud_counts.iterrows():
        quotas[int(row["is_fraud"])] = max(1, int(round(FRAUD_SAMPLE_ROWS * int(row["n"]) / total)))
    # Keep the file at 20,000 rows. The genuine quota absorbs the rounding.
    quotas[0] = FRAUD_SAMPLE_ROWS - quotas.get(1, 0)
    frames = []
    for label, quota in quotas.items():
        part = connection.execute(
            """
            SELECT *
            FROM marts.mart_fraud_transactions
            WHERE is_fraud = ?
            ORDER BY hash(txn_id)
            LIMIT ?
            """,
            [label, quota],
        ).fetchdf()
        frames.append(part)
    fraud = pd.concat(frames, ignore_index=True)
    credit.to_parquet(data_dir / "credit_applications_sample.parquet", index=False)
    fraud.to_parquet(data_dir / "fraud_transactions_sample.parquet", index=False)
    _dump(
        data_dir / "samples.json",
        {
            "credit_rows": int(len(credit)),
            "credit_by_outcome": {
                str(status): int(count)
                for status, count in credit["outcome_status"].value_counts().items()
            },
            "fraud_rows": int(len(fraud)),
            "fraud_rows_in_sample": int(fraud["is_fraud"].sum()),
        },
    )
    print(
        f"Samples: {len(credit):,} applications "
        f"({credit['outcome_status'].value_counts().to_dict()}), "
        f"{len(fraud):,} transactions ({int(fraud['is_fraud'].sum()):,} fraud)"
    )


def _write_decisions(db_path: Path, data_dir: Path) -> dict:
    """Champion decision mix. Reasons come from scoring the mart with credit_v1."""

    connection = duckdb.connect(str(db_path), read_only=True)
    try:
        frame = connection.execute(
            """
            SELECT
                application_id, age, income, bureau_score, debt_to_income,
                num_delinquencies_24m, num_enquiries_6m, utilisation,
                total_credit_limit, requested_amount
            FROM marts.mart_credit_applications
            """
        ).fetchdf()
    finally:
        connection.close()
    decided = decide_frame(frame, load_strategy("credit_v1"), load_model("credit_lr_v1"))
    counts = decided["decision"].value_counts()
    total = len(decided)
    mix = [
        {"decision": name, "count": int(counts.get(name, 0)), "rate": float(counts.get(name, 0) / total)}
        for name in ("APPROVE", "REFER", "DECLINE")
    ]
    declined = decided["decision"].eq("DECLINE")
    reason_rows: list[str] = []
    for reasons in decided.loc[declined, "reason_codes"]:
        reason_rows.extend(reasons)
    top = pd.Series(reason_rows).value_counts().head(8)
    payload = {
        "strategy_version": "credit_v1",
        "model_version": "credit_lr_v1",
        "applications": total,
        "decisions": mix,
        "top_decline_reasons": [
            {"reason": reason, "count": int(count)} for reason, count in top.items()
        ],
        "decline_reason_note": (
            "One decline can carry more than one reason, so the reason counts "
            "add up to more than the number of declined applications."
        ),
    }
    _dump(data_dir / "decision_mix.json", payload)
    print("Credit v1 mix:", ", ".join(f"{row['decision']} {row['count']:,}" for row in mix))
    return payload


def _write_models_and_strategy(root: Path, data_dir: Path, decisions: dict, golden: dict) -> None:
    credit_meta = load_metadata("credit_lr_v1")
    fraud_meta = load_metadata("fraud_hgb_v1")
    _dump(
        data_dir / "credit_model.json",
        {
            "model_version": credit_meta["model_version"],
            "model_type": credit_meta["model_type"],
            "split": credit_meta["split"],
            "maturity_cutoff": credit_meta["maturity_cutoff"],
            "n_train": credit_meta["n_train"],
            "n_test": credit_meta["n_test"],
            "bad_rate_train": credit_meta["bad_rate_train"],
            "bad_rate_test": credit_meta["bad_rate_test"],
            "metrics": credit_meta["metrics"],
            "calibration_deciles": credit_meta["calibration_deciles"],
            "coefficients": credit_meta["coefficients"],
        },
    )
    features = json.loads((data_dir / "fraud_features.json").read_text(encoding="utf-8"))
    _dump(
        data_dir / "fraud_model.json",
        {
            "model_version": fraud_meta["model_version"],
            "model_type": fraud_meta["model_type"],
            "n_train": fraud_meta["n_train"],
            "n_test": fraud_meta["n_test"],
            "fraud_rate_train": fraud_meta["fraud_rate_train"],
            "fraud_rate_test": fraud_meta["fraud_rate_test"],
            "metrics": fraud_meta["metrics"],
            "threshold_table": fraud_meta["threshold_table"],
            "feature_averages": features["by_label"],
        },
    )
    credit_compare = json.loads((root / "reports" / "credit_compare.json").read_text(encoding="utf-8"))
    fraud_compare = json.loads((root / "reports" / "fraud_compare.json").read_text(encoding="utf-8"))
    scored_approves = next(row["count"] for row in decisions["decisions"] if row["decision"] == "APPROVE")
    compared_approves = credit_compare["versions"]["credit_v1"]["approve_count"]
    if scored_approves != compared_approves:
        raise RuntimeError(
            f"Fresh credit_v1 approvals are {scored_approves:,} but the validation "
            f"file has {compared_approves:,}. Re-run python -m src.validation."
        )
    recommendation = recommend(credit_compare, fraud_compare, golden)
    _dump(
        data_dir / "strategy_change.json",
        {
            "recommendation": recommendation,
            "credit": {
                "n_applications": credit_compare["n_applications"],
                "lgd": credit_compare["lgd"],
                "lgd_meaning": credit_compare["lgd_meaning"],
                "swap_matrix": credit_compare["swap_matrix"],
                "swap_ins": credit_compare["swap_ins"],
                "swap_outs": credit_compare["swap_outs"],
                "versions": credit_compare["versions"],
                "swap_causes": credit_compare["swap_causes"],
                "referral_change": credit_compare["referral_change"],
                "bureau_580_to_599": credit_compare["bureau_580_to_599"],
            },
            "fraud": {
                "n_transactions": fraud_compare["n_transactions"],
                "versions": fraud_compare["versions"],
                "fraud_rate_shift": fraud_compare["fraud_rate_shift"],
            },
        },
    )
    _dump(data_dir / "golden.json", golden)
    for name in ("strategy_change_v1_to_v2.md", "strategy_change_v1_to_v2.html"):
        shutil.copy2(root / "reports" / name, data_dir / name)


def _copy_runtime(root: Path) -> None:
    model_dest = root / "app" / "models"
    if model_dest.exists():
        shutil.rmtree(model_dest)
    shutil.copytree(root / "artifacts" / "models", model_dest)
    strategy_dest = root / "app" / "strategies"
    if strategy_dest.exists():
        shutil.rmtree(strategy_dest)
    strategy_dest.mkdir(parents=True)
    for name in STRATEGY_FILES:
        shutil.copy2(root / "strategies" / name, strategy_dest / name)
    shutil.copytree(root / "strategies" / "schema", strategy_dest / "schema")


def _golden(root: Path) -> dict:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_golden.py", "-q", "--tb=line"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    summary = ""
    for line in reversed(output.splitlines()):
        if line.strip():
            summary = line.strip()
            break
    cases = json.loads((root / "tests" / "golden" / "cases.json").read_text(encoding="utf-8"))
    print(output)
    return {
        "passed": completed.returncode == 0,
        "exit_code": completed.returncode,
        "summary": summary,
        "cases": len(cases),
    }


def _check_sizes(root: Path) -> None:
    app = root / "app"
    data_bytes = 0
    largest = 0
    largest_name = ""
    files = []
    for path in app.rglob("*"):
        if not path.is_file():
            continue
        size = path.stat().st_size
        files.append({"path": str(path.relative_to(app)).replace("\\", "/"), "bytes": size})
        largest = max(largest, size)
        if size > largest:
            largest_name = path.name
        if size == largest:
            largest_name = str(path.relative_to(app))
        if path.is_relative_to(app / "data"):
            data_bytes += size
        if size > FILE_BUDGET_BYTES:
            raise RuntimeError(f"{path} is {size:,} bytes, over the 50 MB file limit.")
    if data_bytes > DATA_BUDGET_BYTES:
        raise RuntimeError(f"app/data is {data_bytes:,} bytes, over the 25 MB snapshot limit.")
    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": "2026-10-01",
        "warehouse": "data/warehouse.duckdb",
        "sklearn": _sklearn_version(),
        "models": ["credit_lr_v1", "fraud_hgb_v1"],
        "strategies": ["credit_v1", "credit_v2", "fraud_v1", "fraud_v2"],
        "data_bytes": data_bytes,
        "largest_file": largest_name.replace("\\", "/"),
        "largest_file_bytes": largest,
        "files": files,
        "note": "All rows are synthetic. The snapshot is a sample plus summaries, not the warehouse.",
    }
    _dump(app / "data" / "manifest.json", manifest)
    # The manifest itself counts toward the budget.
    data_bytes = sum(path.stat().st_size for path in (app / "data").rglob("*") if path.is_file())
    if data_bytes > DATA_BUDGET_BYTES:
        raise RuntimeError(f"app/data including the manifest is {data_bytes:,} bytes.")
    manifest["data_bytes"] = data_bytes
    _dump(app / "data" / "manifest.json", manifest)
    print(f"app/data {data_bytes / 1024 / 1024:.2f} MB. Largest file {largest_name} {largest / 1024:.1f} KB.")


def _sklearn_version() -> str:
    import sklearn

    return sklearn.__version__


def _dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_number), encoding="utf-8")


def _number(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def main() -> None:
    export_demo()


if __name__ == "__main__":
    main()
