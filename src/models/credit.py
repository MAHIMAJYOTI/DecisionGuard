"""Logistic regression for the probability of 90+ DPD within 12 months.

Who is in the training sample
-----------------------------
Only booked loans with a known outcome. `not_booked` has no repayment history,
so there is nothing to learn from. `not_yet_known` is a loan that is still too
young: a blank flag is "not yet", not "good".

This sample is biased, and the bias has a name: reject inference.
The old booking rules (age, score, DTI, delinquencies, income) already turned
away the riskier applicants. We never see whether those people would have
reached 90+ DPD. The model therefore describes default risk among people the
old policy was willing to book, not among everyone who applies. Scoring a
declined applicant with it is an extrapolation. The strategy's hard rules
still run first, so the model is not asked to approve someone the policy
already forbids.

A second bias sits in the dates. A loan is labelled good only after 12 months.
A loan can be labelled bad from month 3, once 90+ DPD has already happened.
Training on every "known" row therefore puts early defaults, and almost no
goods, into the newest part of the sample. The saved model does not do that.
It keeps only applications whose 12-month window is already complete
(application date on or before the as-of date minus 12 months) and splits
out-of-time inside that matured cohort. The training report still fits the
old sample once, so the two bad rates can be compared.

Features are the drivers in the simulator's default curve: bureau score, DTI,
delinquencies, utilisation, enquiries. Age is not in the regression because
the simulator treats both very young and older customers as riskier, so one
straight line would not have a single business sign. Income is a policy rule
(and already sits inside DTI and the bureau score) rather than a sixth coefficient.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_gen.config import AS_OF
from src.models.metrics import auc_score, calibration_deciles, gini_score, ks_score
from src.models.store import SavedModel, save_model
from src.warehouse.load import default_db_path

MODEL_VERSION = "credit_lr_v1"

# Sign we expect on the original-unit coefficient of default odds.
# negative: a larger value should lower the chance of 90+ DPD.
FEATURE_SIGNS: dict[str, str] = {
    "bureau_score": "negative",
    "debt_to_income": "positive",
    "num_delinquencies_24m": "positive",
    "utilisation": "positive",
    "num_enquiries_6m": "positive",
}

TRAIN_FRACTION = 0.80
# A 12-month outcome is complete only when the application itself is at least
# 12 months before the frozen as-of date. Disbursement is a few days later,
# so this is slightly stricter than "months_on_book >= 12".
MATURITY_MONTHS = 12


def load_training_frame(db_path=None) -> pd.DataFrame:
    path = db_path or default_db_path()
    connection = duckdb.connect(str(path), read_only=True)
    try:
        frame = connection.execute(
            """
            SELECT
                application_id,
                application_ts,
                bureau_score,
                debt_to_income,
                num_delinquencies_24m,
                utilisation,
                num_enquiries_6m,
                outcome_status
            FROM marts.mart_credit_applications
            WHERE outcome_status IN ('good', 'bad')
            """
        ).fetchdf()
    finally:
        connection.close()
    return frame


def maturity_cutoff() -> pd.Timestamp:
    """Last application date whose 12-month window has finished."""

    return pd.Timestamp(AS_OF) - pd.DateOffset(months=MATURITY_MONTHS)


def train_credit_model(db_path=None) -> dict:
    """Fit the matured model, and print it next to the old unmatured split.

    The old split is fitted only so the comparison is from this same run.
    The file that gets saved is the matured model.
    """

    frame = load_training_frame(db_path)
    features = list(FEATURE_SIGNS)
    before = len(frame)
    frame = frame.dropna(subset=features).copy()
    dropped = before - len(frame)
    frame["application_ts"] = pd.to_datetime(frame["application_ts"])
    frame["label"] = (frame["outcome_status"] == "bad").astype(int)
    cutoff = maturity_cutoff()
    mature = frame[frame["application_ts"].dt.floor("D") <= cutoff].copy()
    old = _fit_split(
        frame,
        "oldest 80% of every known outcome, including early defaults",
    )
    new = _fit_split(
        mature,
        "oldest 80% of applications on or before the as-of date minus 12 months",
    )
    new.update(
        {
            "model_version": MODEL_VERSION,
            "model_type": "logistic_regression",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "target": "dpd90_within_12m among matured booked loans",
            "features": features,
            "dropped_rows_missing_features": int(dropped),
            "maturity_cutoff": cutoff.date().isoformat(),
            "as_of_date": AS_OF,
            "n_known_outcome": int(len(frame)),
            "n_matured": int(len(mature)),
            "previous_unmatured_split": {
                key: old[key]
                for key in (
                    "n_train",
                    "n_test",
                    "train_start",
                    "train_end",
                    "test_start",
                    "test_end",
                    "bad_rate_train",
                    "bad_rate_test",
                    "metrics",
                    "calibration_deciles",
                    "split",
                )
            },
        }
    )
    new["sign_warnings"] = _sign_warnings(new["coefficients"])
    save_model(SavedModel(MODEL_VERSION, "credit", features, new.pop("pipeline")), new)
    _print_comparison(old, new)
    return new


def _fit_split(frame: pd.DataFrame, split_note: str) -> dict:
    features = list(FEATURE_SIGNS)
    ordered = frame.sort_values(["application_ts", "application_id"], kind="mergesort")
    cut = int(len(ordered) * TRAIN_FRACTION)
    train = ordered.iloc[:cut]
    test = ordered.iloc[cut:]
    if train["label"].nunique() < 2 or test["label"].nunique() < 2:
        raise RuntimeError("Train and test each need both good and bad loans.")
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=500, solver="lbfgs")),
        ]
    )
    pipeline.fit(train[features], train["label"])
    test_pd = pipeline.predict_proba(test[features])[:, 1]
    labels = test["label"].to_numpy()
    auc = auc_score(labels, test_pd)
    return {
        "pipeline": pipeline,
        "split": split_note,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_start": _stamp(train["application_ts"].iloc[0]),
        "train_end": _stamp(train["application_ts"].iloc[-1]),
        "test_start": _stamp(test["application_ts"].iloc[0]),
        "test_end": _stamp(test["application_ts"].iloc[-1]),
        "bad_rate_train": float(train["label"].mean()),
        "bad_rate_test": float(test["label"].mean()),
        "metrics": {"auc": auc, "gini": gini_score(auc), "ks": ks_score(labels, test_pd)},
        "coefficients": _coefficients(pipeline, features),
        "calibration_deciles": calibration_deciles(labels, test_pd),
    }


def _print_comparison(old: dict, new: dict) -> None:
    print(f"Credit model {MODEL_VERSION}")
    print(f"  Maturity cutoff (application date): {new['maturity_cutoff']}")
    print(f"  Known-outcome rows: {new['n_known_outcome']:,}   matured rows: {new['n_matured']:,}")
    print(f"  {'':<22} {'old split':>14} {'matured split':>16}")
    print(
        f"  {'train bad rate':<22} {old['bad_rate_train']:>14.4f} {new['bad_rate_train']:>16.4f}"
    )
    print(
        f"  {'test bad rate':<22} {old['bad_rate_test']:>14.4f} {new['bad_rate_test']:>16.4f}"
    )
    for name in ("auc", "gini", "ks"):
        print(
            f"  {name:<22} {old['metrics'][name]:>14.4f} {new['metrics'][name]:>16.4f}"
        )
    print(
        f"  Old train {old['train_start']} to {old['train_end']}  "
        f"n={old['n_train']:,}  test n={old['n_test']:,}"
    )
    print(
        f"  New train {new['train_start']} to {new['train_end']}  "
        f"n={new['n_train']:,}  test n={new['n_test']:,}"
    )
    print("  Coefficients of the saved (matured) model")
    print(f"  {'feature':<28} {'coef':>12} {'odds_ratio':>12} {'expected':>10}")
    for row in new["coefficients"]:
        print(
            f"  {row['feature']:<28} {row['coefficient']:>12.6f} "
            f"{row['odds_ratio']:>12.4f} {row['expected_sign']:>10}"
        )
    for title, block in (("Old calibration", old), ("Matured calibration", new)):
        print(f"  {title} (1 = lowest predicted PD)")
        print(f"  {'decile':>6} {'n':>8} {'predicted':>12} {'actual':>12}")
        for row in block["calibration_deciles"]:
            print(
                f"  {row['decile']:>6} {row['n']:>8,} "
                f"{row['predicted_pd']:>12.4f} {row['actual_bad_rate']:>12.4f}"
            )


def _coefficients(pipeline: Pipeline, features: list[str]) -> list[dict[str, float | str]]:
    """Odds ratios per original unit, not per standard deviation.

    The regression is fitted on standardised columns so income-sized numbers
    and delinquency counts can share one model. Dividing by the training
    standard deviation puts the coefficient back into 'per bureau point'
    or 'per extra delinquency'.
    """

    scaler: StandardScaler = pipeline.named_steps["scaler"]
    model: LogisticRegression = pipeline.named_steps["model"]
    scale = scaler.scale_
    rows: list[dict[str, float | str]] = []
    for index, name in enumerate(features):
        coef = float(model.coef_[0, index] / scale[index])
        rows.append(
            {
                "feature": name,
                "coefficient": coef,
                "odds_ratio": float(np.exp(coef)),
                "expected_sign": FEATURE_SIGNS[name],
            }
        )
    return rows


def _sign_warnings(coefficients: list[dict[str, float | str]]) -> list[str]:
    warnings: list[str] = []
    for row in coefficients:
        coef = float(row["coefficient"])
        expected = str(row["expected_sign"])
        wrong = (expected == "negative" and coef > 0) or (expected == "positive" and coef < 0)
        if not wrong:
            continue
        direction = "lower" if expected == "negative" else "raise"
        message = (
            f"WARNING: {row['feature']} coefficient is {coef:.6f}; "
            f"expected {expected} (a higher value should {direction} default odds)."
        )
        warnings.append(message)
        print(message)
    return warnings


def _stamp(value) -> str:
    return pd.Timestamp(value).isoformat()
