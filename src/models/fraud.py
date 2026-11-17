"""Gradient boosting for the probability a card transaction is fraud.

The split is by time: older transactions train the model, the most recent
ones are the test. A random split would let a later fraud pattern leak into
the training rows of the same customer.

Fraud is about half a percent of transactions. Without a weight, the model
can look accurate by calling almost everything genuine. `class_weight='balanced'`
up-weights the fraud rows so the model has to pay attention to them. The
cost of that choice shows up in the threshold table: a low cutoff catches
more fraud and also stops more genuine customers. A high cutoff does the
opposite. There is no cutoff that does both. The strategy file picks two
operating points; this module only measures them.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

# Set before scikit-learn imports joblib. On this PC joblib cannot see the
# physical core count and would print a long warning on every score.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.models.metrics import auc_score, pr_auc_score, threshold_table
from src.models.store import SavedModel, save_model

MODEL_VERSION = "fraud_hgb_v1"
TRAIN_FRACTION = 0.80

# Columns the mart already has. has_prior_30d is added below because a
# customer's first transactions have no 30-day average.
RAW_FEATURES = [
    "amount",
    "txn_count_1h",
    "txn_amount_1h",
    "txn_count_24h",
    "txn_amount_24h",
    "avg_amount_30d",
    "amount_vs_avg_30d",
    "is_new_device",
    "is_foreign_country",
    "is_night",
    "has_prior_30d",
]

# Cutoffs printed with the test metrics. The strategy's own cutoffs are added
# when the report is reprinted, so the table can be read next to the policy.
REPORT_THRESHOLDS = [0.05, 0.20, 0.30, 0.35, 0.40, 0.50, 0.80]


def prepare_fraud_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill a missing 30-day history so the model can score a first transaction.

    The rule engine does not use this fill. A missing ratio is not treated as
    'five times average'. Only the model sees has_prior_30d = 0 and a ratio of 1.
    """

    prepared = frame.copy()
    if "avg_amount_30d" not in prepared.columns:
        prepared["avg_amount_30d"] = np.nan
    if "amount_vs_avg_30d" not in prepared.columns:
        prepared["amount_vs_avg_30d"] = np.nan
    prepared["has_prior_30d"] = prepared["avg_amount_30d"].notna().astype(int)
    prepared["avg_amount_30d"] = prepared["avg_amount_30d"].fillna(0.0)
    prepared["amount_vs_avg_30d"] = prepared["amount_vs_avg_30d"].fillna(1.0)
    return prepared


def load_training_frame(db_path=None) -> pd.DataFrame:
    import duckdb

    from src.warehouse.load import default_db_path

    path = db_path or default_db_path()
    connection = duckdb.connect(str(path), read_only=True)
    try:
        frame = connection.execute(
            """
            SELECT
                txn_id,
                txn_ts,
                amount,
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
    return frame


def train_fraud_model(db_path=None) -> dict:
    frame = prepare_fraud_frame(load_training_frame(db_path))
    frame["label"] = frame["is_fraud"].astype(int)
    frame = frame.sort_values(["txn_ts", "txn_id"], kind="mergesort")
    cut = int(len(frame) * TRAIN_FRACTION)
    train = frame.iloc[:cut]
    test = frame.iloc[cut:]
    if train["label"].nunique() < 2 or test["label"].nunique() < 2:
        raise RuntimeError("Train and test each need both fraud and genuine rows.")

    model = HistGradientBoostingClassifier(
        max_iter=80,
        learning_rate=0.08,
        max_depth=6,
        min_samples_leaf=50,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=8,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(train[RAW_FEATURES], train["label"])
    test_pd = model.predict_proba(test[RAW_FEATURES])[:, 1]
    labels = test["label"].to_numpy()
    # Precision and recall at 0.5, the model's own default cutoff, plus the table.
    at_half = threshold_table(labels, test_pd, [0.5])[0]
    table = threshold_table(labels, test_pd, REPORT_THRESHOLDS)
    metadata = {
        "model_version": MODEL_VERSION,
        "model_type": "hist_gradient_boosting",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "target": "is_fraud",
        "class_weight": "balanced",
        "features": RAW_FEATURES,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_start": _stamp(train["txn_ts"].iloc[0]),
        "train_end": _stamp(train["txn_ts"].iloc[-1]),
        "test_start": _stamp(test["txn_ts"].iloc[0]),
        "test_end": _stamp(test["txn_ts"].iloc[-1]),
        "fraud_rate_train": float(train["label"].mean()),
        "fraud_rate_test": float(test["label"].mean()),
        "metrics": {
            "auc": auc_score(labels, test_pd),
            "pr_auc": pr_auc_score(labels, test_pd),
            "precision_at_0_5": at_half["precision"],
            "recall_at_0_5": at_half["fraud_caught"],
        },
        "threshold_table": table,
        "split": "oldest 80% of transactions train, newest 20% test",
        "tradeoff": (
            "A lower threshold catches more fraud and also challenges more "
            "genuine customers. A higher threshold does the reverse. "
            "Precision is the share of alerts that are real fraud."
        ),
    }
    save_model(SavedModel(MODEL_VERSION, "fraud", RAW_FEATURES, model), metadata)
    _print_report(metadata)
    return metadata


def _print_report(metadata: dict) -> None:
    metrics = metadata["metrics"]
    print(f"Fraud model {metadata['model_version']}")
    print(
        f"  Train {metadata['train_start']} to {metadata['train_end']}  "
        f"n={metadata['n_train']:,}  fraud rate={metadata['fraud_rate_train']:.4f}"
    )
    print(
        f"  Test  {metadata['test_start']} to {metadata['test_end']}  "
        f"n={metadata['n_test']:,}  fraud rate={metadata['fraud_rate_test']:.4f}"
    )
    print(f"  AUC       {metrics['auc']:.4f}")
    print(f"  PR-AUC    {metrics['pr_auc']:.4f}")
    print(f"  Precision at 0.50  {metrics['precision_at_0_5']:.4f}")
    print(f"  Recall at 0.50     {metrics['recall_at_0_5']:.4f}")
    print(f"  {metadata['tradeoff']}")
    print("  Threshold table (test set)")
    print(
        f"  {'threshold':>10} {'alert_rate':>12} {'precision':>10} "
        f"{'fraud_caught':>13} {'false_pos':>12}"
    )
    for row in metadata["threshold_table"]:
        print(
            f"  {row['threshold']:>10.2f} {row['alert_rate']:>12.4f} "
            f"{row['precision']:>10.4f} {row['fraud_caught']:>13.4f} "
            f"{int(row['false_positives']):>12,}"
        )


def _stamp(value) -> str:
    return pd.Timestamp(value).isoformat()
