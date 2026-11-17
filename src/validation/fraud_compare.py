"""Score the same transactions with fraud v1 and fraud v2.

Alert means the decision is CHALLENGE or BLOCK. A genuine grocery burst is
five genuine grocery tickets for one customer spanning at most 20 minutes,
which is how the generator plants the busy-hour false positive.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from src.data_gen.config import TRANSACTION_START, project_root
from src.decision_engine.engine import decide_frame
from src.decision_engine.strategy import load_strategy
from src.models.fraud import TRAIN_FRACTION
from src.models.store import load_model
from src.warehouse.load import default_db_path


def compare_fraud(db_path: Path | None = None) -> dict:
    path = db_path or default_db_path()
    model = load_model("fraud_hgb_v1")
    frame = _load(path)
    champion = decide_frame(frame, load_strategy("fraud_v1"), model)
    challenger = decide_frame(frame, load_strategy("fraud_v2"), model)
    bursts = _grocery_bursts(frame)
    result = {
        "model_version": model.version,
        "n_transactions": int(len(frame)),
        "fraud_rate_shift": fraud_rate_shift(frame),
        "versions": {
            "fraud_v1": _metrics(frame, champion, bursts),
            "fraud_v2": _metrics(frame, challenger, bursts),
        },
    }
    _print(result)
    return result


def fraud_rate_shift(frame: pd.DataFrame) -> dict:
    """Why the time-split fraud rate is higher in the test window.

    Legitimate tickets are spread over all 90 days. Fraud tickets are placed
    only on day 40 or later, so that a 30-day history can exist. The oldest
    80% of rows therefore include a long quiet period with almost no fraud.
    The newest 20% sits entirely inside the window where fraud was planted.
    """

    ordered = frame.sort_values(["txn_ts", "txn_id"], kind="mergesort")
    cut = int(len(ordered) * TRAIN_FRACTION)
    train = ordered.iloc[:cut]
    test = ordered.iloc[cut:]
    day_40 = pd.Timestamp(TRANSACTION_START) + pd.Timedelta(days=40)
    early = ordered[ordered["txn_ts"] < day_40]
    late = ordered[ordered["txn_ts"] >= day_40]
    return {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_fraud_rate": float(train["is_fraud"].mean()),
        "test_fraud_rate": float(test["is_fraud"].mean()),
        "train_end": str(pd.Timestamp(train["txn_ts"].iloc[-1])),
        "test_start": str(pd.Timestamp(test["txn_ts"].iloc[0])),
        "fraud_start_day": str(day_40.date()),
        "rows_before_fraud_window": int(len(early)),
        "frauds_before_fraud_window": int(early["is_fraud"].sum()),
        "rows_in_fraud_window": int(len(late)),
        "frauds_in_fraud_window": int(late["is_fraud"].sum()),
        "fraud_rate_in_fraud_window": float(late["is_fraud"].mean()) if len(late) else None,
    }


def _load(path: Path) -> pd.DataFrame:
    connection = duckdb.connect(str(path), read_only=True)
    try:
        frame = connection.execute(
            """
            SELECT
                txn_id,
                customer_id,
                amount,
                merchant_category,
                txn_ts,
                txn_count_1h,
                txn_amount_1h,
                txn_count_24h,
                txn_amount_24h,
                avg_amount_30d,
                amount_vs_avg_30d,
                is_new_device,
                is_foreign_country,
                is_night,
                is_fraud
            FROM marts.mart_fraud_transactions
            """
        ).fetchdf()
    finally:
        connection.close()
    frame["txn_ts"] = pd.to_datetime(frame["txn_ts"])
    return frame.reset_index(drop=True)


def _grocery_bursts(frame: pd.DataFrame) -> pd.Series:
    """Mark genuine grocery tickets that sit in a 5-ticket, 20-minute run."""

    flag = pd.Series(False, index=frame.index)
    grocery = frame[(frame["merchant_category"] == "grocery") & (frame["is_fraud"] == 0)].sort_values(
        ["customer_id", "txn_ts"]
    )
    if grocery.empty:
        return flag
    grocery = grocery.copy()
    grocery["pos"] = grocery.groupby("customer_id").cumcount()
    grocery["span"] = grocery["txn_ts"] - grocery.groupby("customer_id")["txn_ts"].shift(4)
    ends = grocery[grocery["span"].notna() & (grocery["span"] <= pd.Timedelta(minutes=20))]
    lookup = {
        (customer_id, int(pos)): idx
        for idx, customer_id, pos in zip(grocery.index, grocery["customer_id"], grocery["pos"], strict=True)
    }
    marked: set[int] = set()
    for customer_id, pos in zip(ends["customer_id"], ends["pos"], strict=True):
        for back in range(5):
            marked.add(lookup[(customer_id, int(pos) - back)])
    if marked:
        flag.loc[list(marked)] = True
    return flag


def _metrics(frame: pd.DataFrame, decided: pd.DataFrame, bursts: pd.Series) -> dict:
    decision = decided["decision"]
    alert = decision.isin(["CHALLENGE", "BLOCK"])
    fraud = frame["is_fraud"].astype(int).eq(1)
    genuine = ~fraud
    caught = fraud & alert
    blocked = decision.eq("BLOCK")
    challenged = decision.eq("CHALLENGE")
    velocity = (
        genuine
        & frame["merchant_category"].eq("grocery")
        & (pd.to_numeric(frame["txn_count_1h"], errors="coerce") >= 3)
    )
    return {
        "alert_rate": float(alert.mean()),
        "fraud_caught_rate": float(caught.sum() / fraud.sum()) if fraud.any() else None,
        "fraud_value_caught": float(frame.loc[caught, "amount"].sum()),
        "fraud_value_blocked": float(frame.loc[fraud & blocked, "amount"].sum()),
        "fraud_value_challenged": float(frame.loc[fraud & challenged, "amount"].sum()),
        "false_positives": int((genuine & alert).sum()),
        "genuine_transactions_blocked": int((genuine & blocked).sum()),
        "genuine_transactions_challenged": int((genuine & challenged).sum()),
        "genuine_customers_blocked": int(frame.loc[genuine & blocked, "customer_id"].nunique()),
        "genuine_customers_challenged": int(frame.loc[genuine & challenged, "customer_id"].nunique()),
        "grocery_burst_transactions": int(bursts.sum()),
        "grocery_burst_blocked": int((bursts & blocked).sum()),
        "grocery_burst_challenged": int((bursts & challenged).sum()),
        "grocery_burst_allowed": int((bursts & decision.eq("ALLOW")).sum()),
        "genuine_grocery_velocity_rows": int(velocity.sum()),
        "genuine_grocery_velocity_blocked": int((velocity & blocked).sum()),
        "genuine_grocery_velocity_challenged": int((velocity & challenged).sum()),
    }


def _print(result: dict) -> None:
    shift = result["fraud_rate_shift"]
    print("Fraud rate shift (time split on the scored mart)")
    print(
        f"  Train fraud rate {shift['train_fraud_rate']:.4f} on {shift['train_rows']:,} rows "
        f"ending {shift['train_end']}"
    )
    print(
        f"  Test fraud rate  {shift['test_fraud_rate']:.4f} on {shift['test_rows']:,} rows "
        f"starting {shift['test_start']}"
    )
    print(
        f"  Before {shift['fraud_start_day']}: {shift['frauds_before_fraud_window']:,} frauds "
        f"in {shift['rows_before_fraud_window']:,} rows"
    )
    print(
        f"  From {shift['fraud_start_day']}: {shift['frauds_in_fraud_window']:,} frauds "
        f"in {shift['rows_in_fraud_window']:,} rows "
        f"(rate {shift['fraud_rate_in_fraud_window']:.4f})"
    )
    print("Fraud strategy comparison")
    rows = [
        ("alert_rate", "rate"),
        ("fraud_caught_rate", "rate"),
        ("fraud_value_caught", "rupees"),
        ("fraud_value_blocked", "rupees"),
        ("fraud_value_challenged", "rupees"),
        ("false_positives", "count"),
        ("genuine_transactions_blocked", "count"),
        ("genuine_transactions_challenged", "count"),
        ("genuine_customers_blocked", "count"),
        ("genuine_customers_challenged", "count"),
        ("grocery_burst_transactions", "count"),
        ("grocery_burst_blocked", "count"),
        ("grocery_burst_challenged", "count"),
        ("grocery_burst_allowed", "count"),
        ("genuine_grocery_velocity_rows", "count"),
        ("genuine_grocery_velocity_blocked", "count"),
        ("genuine_grocery_velocity_challenged", "count"),
    ]
    print(f"  {'metric':<36} {'fraud_v1':>16} {'fraud_v2':>16}")
    for key, kind in rows:
        left = result["versions"]["fraud_v1"][key]
        right = result["versions"]["fraud_v2"][key]
        print(f"  {key:<36} {_cell(left, kind):>16} {_cell(right, kind):>16}")


def _cell(value: float | int | None, kind: str) -> str:
    if value is None:
        return "n/a"
    if kind == "rate":
        return f"{float(value):.4f}"
    if kind == "rupees":
        return f"{float(value):,.0f}"
    return f"{int(value):,}"


def main() -> None:
    result = compare_fraud()
    out = project_root() / "reports" / "fraud_compare.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
