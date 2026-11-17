"""Scores we report for the two models.

AUC is the chance the model ranks a bad case above a good one.
Gini rescales AUC so that 0 means no skill and 1 means a perfect ranking.
KS is the widest gap between the cumulative share of bads and the cumulative
share of goods as you walk down the score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))


def gini_score(auc: float) -> float:
    return 2.0 * auc - 1.0


def ks_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Maximum gap between the bad and good cumulative curves."""

    order = np.argsort(-y_score, kind="mergesort")
    labels = np.asarray(y_true)[order]
    n_bad = float(labels.sum())
    n_good = float(len(labels) - n_bad)
    if n_bad == 0 or n_good == 0:
        raise ValueError("KS needs at least one bad and one good row.")
    cum_bad = np.cumsum(labels) / n_bad
    cum_good = np.cumsum(1.0 - labels) / n_good
    return float(np.max(np.abs(cum_bad - cum_good)))


def calibration_deciles(y_true: np.ndarray, y_score: np.ndarray) -> list[dict[str, float]]:
    """Ten groups from lowest predicted probability to highest.

    Each row is the average prediction next to the share that actually went bad.
    A well calibrated model has those two numbers close together.
    """

    frame = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_score)})
    frame["decile"] = pd.qcut(frame["p"], 10, labels=False, duplicates="drop") + 1
    rows: list[dict[str, float]] = []
    for decile, group in frame.groupby("decile", sort=True):
        rows.append(
            {
                "decile": int(decile),
                "n": int(len(group)),
                "predicted_pd": float(group["p"].mean()),
                "actual_bad_rate": float(group["y"].mean()),
            }
        )
    return rows


def threshold_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: list[float],
) -> list[dict[str, float]]:
    """What happens if every score at or above a cutoff raises an alert.

    alert_rate is the share of all transactions we would stop and look at.
    fraud_caught is the share of real fraud those alerts contain (recall).
    false_positives is the count of genuine transactions in the alerts.
    """

    labels = np.asarray(y_true).astype(int)
    scores = np.asarray(y_score)
    n = len(labels)
    n_fraud = int(labels.sum())
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        alert = scores >= threshold
        caught = int((alert & (labels == 1)).sum())
        false_positives = int((alert & (labels == 0)).sum())
        alerts = int(alert.sum())
        rows.append(
            {
                "threshold": float(threshold),
                "alert_rate": alerts / n if n else 0.0,
                "precision": (caught / alerts) if alerts else 0.0,
                "fraud_caught": (caught / n_fraud) if n_fraud else 0.0,
                "false_positives": false_positives,
                "alerts": alerts,
            }
        )
    return rows


def pr_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision-recall curve. More honest than ROC when fraud is rare."""

    return float(average_precision_score(y_true, y_score))
