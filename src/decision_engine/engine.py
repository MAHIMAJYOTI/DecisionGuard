"""Turn one application, or a whole table, into a decision.

Order, and it is always this order:

1. Data checks. A missing income or a missing bureau score is REFER.
   The automated path never approves those rows, and it does not turn them
   into a hard decline either: a person has to look, because the field the
   policy wanted is not there.
2. Policy rules. Every hard decline that fires is kept, in the order the
   YAML lists them. Any one of them makes the decision DECLINE.
3. Model score. Probability of 90+ DPD, or of fraud.
4. Decision band. Approve / refer / decline, or allow / challenge / block.
   A thin credit file uses the tighter band. For fraud, a rule and the model
   both vote and the harsher vote wins.
5. Limit. Only an APPROVE gets one. It is the smallest of the amount asked
   for, a fraction of income, and the cap for that bureau band.

The same input and the same strategy and model versions always take the same
path. Nothing here draws a random number.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.decision_engine.strategy import Strategy
from src.models.fraud import prepare_fraud_frame

_OPS = {
    "lt": lambda column, value: column < value,
    "lte": lambda column, value: column <= value,
    "gt": lambda column, value: column > value,
    "gte": lambda column, value: column >= value,
    "eq": lambda column, value: column == value,
}


def decide_one(record: dict, strategy: Strategy, model) -> dict:
    """Score a single application or transaction given as a plain dict."""

    frame = pd.DataFrame([record])
    decided = decide_frame(frame, strategy, model).iloc[0]
    score = decided["score"]
    limit = decided["credit_limit"]
    return {
        "decision": decided["decision"],
        "reason_codes": list(decided["reason_codes"]),
        "score": None if pd.isna(score) else float(score),
        "strategy_version": decided["strategy_version"],
        "model_version": decided["model_version"],
        "credit_limit": None if pd.isna(limit) else float(limit),
    }


def decide_frame(frame: pd.DataFrame, strategy: Strategy, model) -> pd.DataFrame:
    """Score every row. Row i of the result belongs to row i of `frame`."""

    source = frame.reset_index(drop=True)
    count = len(source)
    reasons: list[list[str]] = [[] for _ in range(count)]
    limits = np.full(count, np.nan)
    if strategy.kind == "fraud":
        model_input = prepare_fraud_frame(source)
        scores = np.asarray(model.predict_pd(model_input), dtype=float)
        decisions = _decide_fraud(source, strategy, scores, reasons)
    else:
        scores = np.asarray(model.predict_pd(source), dtype=float)
        decisions = _decide_credit(source, strategy, scores, reasons, limits)
    entity_column = "application_id" if strategy.kind == "credit" else "txn_id"
    if entity_column in source.columns:
        entity_ids = source[entity_column].astype(str).tolist()
    else:
        entity_ids = [f"row-{index}" for index in range(count)]
    return pd.DataFrame(
        {
            "entity_id": entity_ids,
            "decision": decisions,
            "reason_codes": reasons,
            "score": scores,
            "strategy_version": strategy.version,
            "model_version": model.version,
            "credit_limit": limits,
        }
    )


def _decide_credit(
    frame: pd.DataFrame,
    strategy: Strategy,
    scores: np.ndarray,
    reasons: list[list[str]],
    limits: np.ndarray,
) -> np.ndarray:
    count = len(frame)
    decisions = np.empty(count, dtype=object)
    blocked = np.zeros(count, dtype=bool)
    for check in strategy.data_checks:
        missing = _is_missing(frame, check.field)
        blocked |= missing
        _add(reasons, missing, check.reason)
    decisions[blocked] = "REFER"

    declined = np.zeros(count, dtype=bool)
    for rule in strategy.policy_rules:
        # A row already referred for missing data is not also auto-declined.
        hit = _hit(frame, rule.field, rule.op, rule.value) & ~blocked
        declined |= hit
        _add(reasons, hit, rule.reason)
    decisions[declined] = "DECLINE"

    open_rows = ~blocked & ~declined
    thin = np.zeros(count, dtype=bool)
    if strategy.thin_file is not None:
        band = strategy.thin_file
        thin = _hit(frame, band.field, band.op, band.value) & open_rows
    approve_cut = np.where(thin, strategy.thin_file.approve_max_pd, strategy.approve_max_pd)
    refer_cut = np.where(thin, strategy.thin_file.refer_max_pd, strategy.refer_max_pd)
    missing_score = open_rows & np.isnan(scores)
    approve = open_rows & ~missing_score & (scores <= approve_cut)
    refer = open_rows & ~missing_score & ~approve & (scores <= refer_cut)
    decline_score = open_rows & ~missing_score & ~approve & ~refer
    decisions[missing_score] = "REFER"
    decisions[approve] = "APPROVE"
    decisions[refer] = "REFER"
    decisions[decline_score] = "DECLINE"
    _add(reasons, thin & open_rows & ~missing_score, "THIN_FILE")
    _add(reasons, missing_score, "MISSING_MODEL_FEATURES")
    _add(reasons, approve, "SCORE_WITHIN_APPROVE_BAND")
    _add(reasons, refer, "SCORE_WITHIN_REFER_BAND")
    _add(reasons, decline_score, "MODEL_PD_ABOVE_DECLINE_CUTOFF")
    _assign_limits(frame, strategy, decisions, limits)
    return decisions


def _decide_fraud(
    frame: pd.DataFrame,
    strategy: Strategy,
    scores: np.ndarray,
    reasons: list[list[str]],
) -> np.ndarray:
    count = len(frame)
    decisions = np.empty(count, dtype=object)
    blocked = np.zeros(count, dtype=bool)
    for check in strategy.data_checks:
        missing = _is_missing(frame, check.field)
        blocked |= missing
        _add(reasons, missing, check.reason)
    decisions[blocked] = "CHALLENGE"

    # 0 allow, 1 challenge, 2 block. The larger number wins.
    severity = np.zeros(count, dtype=int)
    open_rows = ~blocked
    for rule in strategy.fraud_rules:
        hit = open_rows.copy()
        for condition in rule.when:
            hit &= _hit(frame, condition.field, condition.op, condition.value)
        _add(reasons, hit, rule.reason)
        level = 2 if rule.decision == "BLOCK" else 1
        severity = np.maximum(severity, np.where(hit, level, 0))

    missing_score = open_rows & np.isnan(scores)
    challenge_cut = float(strategy.challenge_min_pd)
    block_cut = float(strategy.block_min_pd)
    model_level = np.zeros(count, dtype=int)
    scored = open_rows & ~missing_score
    model_level[scored & (scores >= block_cut)] = 2
    model_level[scored & (scores >= challenge_cut) & (scores < block_cut)] = 1
    severity = np.maximum(severity, model_level)
    severity = np.maximum(severity, np.where(missing_score, 1, 0))
    _add(reasons, missing_score, "MISSING_MODEL_FEATURES")
    _add(reasons, scored & (model_level == 2), "MODEL_SCORE_BLOCK")
    _add(reasons, scored & (model_level == 1), "MODEL_SCORE_CHALLENGE")
    _add(reasons, scored & (model_level == 0), "MODEL_SCORE_ALLOW")
    decisions[open_rows & (severity == 2)] = "BLOCK"
    decisions[open_rows & (severity == 1)] = "CHALLENGE"
    decisions[open_rows & (severity == 0)] = "ALLOW"
    return decisions


def _assign_limits(
    frame: pd.DataFrame,
    strategy: Strategy,
    decisions: np.ndarray,
    limits: np.ndarray,
) -> None:
    by_band = {str(rule["risk_band"]): rule for rule in strategy.limit_rules}
    income = _numbers(frame, "income")
    requested = _numbers(frame, "requested_amount")
    bureau = _numbers(frame, "bureau_score")
    for index in np.flatnonzero(decisions == "APPROVE"):
        score = float(bureau.iloc[index])
        band_name = _band_name(score, strategy)
        rule = by_band.get(band_name)
        if rule is None or pd.isna(income.iloc[index]):
            continue
        parts = [float(rule["cap"]), float(income.iloc[index]) * float(rule["income_fraction"])]
        if not pd.isna(requested.iloc[index]):
            parts.append(float(requested.iloc[index]))
        limits[index] = _round_rupees(min(parts))


def _band_name(score: float, strategy: Strategy) -> str:
    chosen = str(strategy.risk_bands[-1]["name"])
    for band in strategy.risk_bands:
        if score <= float(band["max_score"]):
            return str(band["name"])
    return chosen


def _round_rupees(amount: float) -> float:
    if amount >= 1000:
        return float(math.floor(amount / 1000.0) * 1000)
    return round(amount, 2)


def _is_missing(frame: pd.DataFrame, field: str) -> np.ndarray:
    """True when the field is absent, null, or blank.

    Timestamps stay text here. Turning them into numbers would mark every
    real transaction time as missing.
    """

    if field not in frame.columns:
        return np.ones(len(frame), dtype=bool)
    column = frame[field]
    missing = column.isna()
    if column.dtype == object:
        blank = column.astype("string").str.strip().isin(["", "None", "nan", "NaT"])
        missing = missing | blank.fillna(False)
    return missing.to_numpy()


def _numbers(frame: pd.DataFrame, field: str) -> pd.Series:
    if field not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[field], errors="coerce")


def _hit(frame: pd.DataFrame, field: str, op: str, value: float) -> np.ndarray:
    column = _numbers(frame, field)
    compared = _OPS[op](column, value)
    return (column.notna() & compared).fillna(False).to_numpy()


def _add(reasons: list[list[str]], mask: np.ndarray, reason: str) -> None:
    for index in np.flatnonzero(mask):
        reasons[index].append(reason)

