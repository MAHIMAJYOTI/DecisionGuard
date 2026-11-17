"""Score a batch from the warehouse, or one application passed as an example.

    python -m src.decision_engine --strategy credit_v1 --batch applications
    python -m src.decision_engine --strategy credit_v1 --example
    python -m src.decision_engine --strategy fraud_v1 --batch transactions
"""

from __future__ import annotations

import argparse
import json
import uuid
from collections import Counter
from pathlib import Path

import duckdb
import pandas as pd

from src.decision_engine.engine import decide_frame, decide_one
from src.decision_engine.log import write_decisions
from src.decision_engine.strategy import StrategyError, load_strategy
from src.models.store import load_model
from src.warehouse.load import default_db_path

# A clean personal-loan application. Numbers are inside every v1 hard rule
# (age 36, score 742, DTI 28%, no delinquencies, income 7.2 lakh) so the
# model and the limit rule are what decide it.
CREDIT_EXAMPLE = {
    "application_id": "EXAMPLE0001",
    "product": "personal loan",
    "age": 36,
    "income": 720_000,
    "bureau_score": 742,
    "debt_to_income": 0.28,
    "num_delinquencies_24m": 0,
    "num_enquiries_6m": 1,
    "utilisation": 0.35,
    "total_credit_limit": 250_000,
    "requested_amount": 150_000,
}

FRAUD_EXAMPLE = {
    "txn_id": "EXAMPLETXN1",
    "amount": 850.0,
    "txn_count_1h": 0,
    "txn_amount_1h": 0.0,
    "txn_count_24h": 1,
    "txn_amount_24h": 640.0,
    "avg_amount_30d": 900.0,
    "amount_vs_avg_30d": 0.94,
    "is_new_device": 0,
    "is_foreign_country": 0,
    "is_night": 0,
    "txn_ts": "2026-09-15 14:10:00",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a versioned credit or fraud strategy.")
    parser.add_argument("--strategy", required=True, help="Strategy version, for example credit_v1.")
    parser.add_argument("--batch", choices=["applications", "transactions"], help="Score a warehouse mart.")
    parser.add_argument("--example", action="store_true", help="Score the built-in single application or transaction.")
    parser.add_argument("--input", type=Path, help="JSON file with one application or transaction object.")
    parser.add_argument("--db-path", type=Path, default=default_db_path())
    args = parser.parse_args()
    if sum(bool(item) for item in (args.batch, args.example, args.input)) != 1:
        parser.error("Choose exactly one of --batch, --example, or --input.")

    try:
        strategy = load_strategy(args.strategy)
    except StrategyError as exc:
        raise SystemExit(str(exc)) from exc
    model = load_model(strategy.model_version)
    if model.version != strategy.model_version:
        raise SystemExit(
            f"Loaded {model.version}, but {strategy.version} requires {strategy.model_version}."
        )

    if args.batch:
        _run_batch(strategy, model, args.batch, args.db_path)
        return
    record = FRAUD_EXAMPLE if strategy.kind == "fraud" else CREDIT_EXAMPLE
    if args.input:
        record = json.loads(args.input.read_text(encoding="utf-8"))
    decision = decide_one(record, strategy, model)
    print("Input")
    for key, value in record.items():
        print(f"  {key}: {value}")
    print("Decision")
    _print_decision(decision)
    if args.db_path.exists():
        run_id = uuid.uuid4().hex[:12]
        written = write_decisions(args.db_path, pd.DataFrame([record]), pd.DataFrame([decision]), run_id)
        print(f"Logged {written} row to decisions.decision_log (run {run_id})")


def _run_batch(strategy, model, batch: str, db_path: Path) -> None:
    if batch == "applications" and strategy.kind != "credit":
        raise SystemExit(f"{strategy.version} is a {strategy.kind} strategy, not a credit strategy.")
    if batch == "transactions" and strategy.kind != "fraud":
        raise SystemExit(f"{strategy.version} is a {strategy.kind} strategy, not a fraud strategy.")
    table = "marts.mart_credit_applications" if batch == "applications" else "marts.mart_fraud_transactions"
    connection = duckdb.connect(str(db_path), read_only=True)
    try:
        frame = connection.execute(f"SELECT * FROM {table}").fetchdf()
    finally:
        connection.close()
    decided = decide_frame(frame, strategy, model)
    run_id = uuid.uuid4().hex[:12]
    written = write_decisions(db_path, frame, decided, run_id)
    _print_mix(decided, strategy, written, run_id)


def _print_decision(decision: dict) -> None:
    print(f"  decision: {decision['decision']}")
    print(f"  reason_codes: {', '.join(decision['reason_codes'])}")
    score = decision["score"]
    print(f"  score: {score:.6f}" if score is not None else "  score: null")
    print(f"  strategy_version: {decision['strategy_version']}")
    print(f"  model_version: {decision['model_version']}")
    if decision.get("credit_limit") is not None:
        print(f"  credit_limit: {decision['credit_limit']:,.0f}")


def _print_mix(decided: pd.DataFrame, strategy, written: int, run_id: str) -> None:
    total = len(decided)
    print(
        f"Scored {total:,} rows with {strategy.version} / {strategy.model_version} "
        f"(logged {written:,}, run {run_id})"
    )
    counts = decided["decision"].value_counts()
    for name in counts.index:
        count = int(counts[name])
        print(f"  {name:<10} {count:>8,}  {100.0 * count / total:6.2f}%")
    if strategy.kind != "credit":
        return
    declined = decided[decided["decision"] == "DECLINE"]
    tally: Counter[str] = Counter()
    for reasons in declined["reason_codes"]:
        tally.update(reasons)
    print("Top decline reasons")
    if not tally:
        print("  (none)")
        return
    for reason, count in tally.most_common(5):
        print(f"  {reason:<32} {count:>8,}")


if __name__ == "__main__":
    main()
