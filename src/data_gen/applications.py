"""Loan and card applications. One row per application.

Customers are drawn with replacement, so some people apply twice and some
never apply. A real queue looks like that; it is not one application per
customer.

`tenure_months` is 0 on a credit card because a card has no fixed term.
That 0 is meaningful, not a missing value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_gen.config import (
    APPLICATION_COLUMNS,
    CHANNELS,
    PRODUCTS,
    DataGenConfig,
    legit_hour_weights,
)
from src.data_gen.utils import round_money

_PERSONAL_LOAN_TENURES = np.array([12, 24, 36, 48, 60])
_PERSONAL_LOAN_TENURE_P = (0.12, 0.28, 0.30, 0.18, 0.12)


def generate_applications(
    rng: np.random.Generator,
    customers: pd.DataFrame,
    cfg: DataGenConfig,
) -> pd.DataFrame:
    """Build the clean application file and sort it by time before assigning ids."""

    n = cfg.n_applications
    picked = rng.choice(len(customers), size=n, replace=True)
    product = rng.choice(list(PRODUCTS), size=n, p=[0.62, 0.38])
    channel = rng.choice(list(CHANNELS), size=n, p=[0.34, 0.41, 0.25])
    timestamps = _timestamps(rng, cfg, channel)

    frame = pd.DataFrame(
        {
            "customer_id": customers["customer_id"].to_numpy()[picked],
            "product": product,
            "requested_amount": _requested_amount(rng, product),
            "tenure_months": _tenure(rng, product),
            "application_ts": timestamps,
            "channel": channel,
        }
    )
    frame = frame.sort_values("application_ts").reset_index(drop=True)
    frame.insert(0, "application_id", [f"APP{i:08d}" for i in range(1, n + 1)])
    frame["application_ts"] = pd.to_datetime(frame["application_ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    _check(frame, customers)
    return frame[list(APPLICATION_COLUMNS)]


def _timestamps(
    rng: np.random.Generator,
    cfg: DataGenConfig,
    channel: np.ndarray,
) -> pd.Series:
    """Branch applications land in opening hours. Web and app can arrive any time."""

    n = len(channel)
    start = pd.Timestamp(cfg.application_start)
    day = rng.integers(0, cfg.application_days, n)
    hour = rng.choice(24, size=n, p=legit_hour_weights())
    branch = channel == "branch"
    n_branch = int(branch.sum())
    if n_branch:
        hour[branch] = rng.integers(10, 17, size=n_branch)
    minute = rng.integers(0, 60, n)
    second = rng.integers(0, 60, n)
    stamps = (
        start
        + pd.to_timedelta(day, unit="D")
        + pd.to_timedelta(hour, unit="h")
        + pd.to_timedelta(minute, unit="m")
        + pd.to_timedelta(second, unit="s")
    )
    return pd.Series(stamps)


def _requested_amount(rng: np.random.Generator, product: np.ndarray) -> np.ndarray:
    """Personal loans centre near 2.5 lakh. Card limits centre near 80 thousand."""

    amount = np.empty(len(product))
    loan = product == "personal loan"
    card = product == "credit card"
    if loan.any():
        drawn = rng.lognormal(12.43, 0.55, int(loan.sum()))
        amount[loan] = np.clip(drawn, 20_000, 2_000_000)
    if card.any():
        drawn = rng.lognormal(11.29, 0.45, int(card.sum()))
        amount[card] = np.clip(drawn, 15_000, 800_000)
    return round_money(amount)


def _tenure(rng: np.random.Generator, product: np.ndarray) -> np.ndarray:
    tenure = np.zeros(len(product), dtype=int)
    loan = product == "personal loan"
    if loan.any():
        tenure[loan] = rng.choice(
            _PERSONAL_LOAN_TENURES,
            size=int(loan.sum()),
            p=_PERSONAL_LOAN_TENURE_P,
        )
    return tenure


def _check(frame: pd.DataFrame, customers: pd.DataFrame) -> None:
    if not frame["application_id"].is_unique:
        raise RuntimeError("clean application ids must be unique")
    if not set(frame["customer_id"]).issubset(set(customers["customer_id"])):
        raise RuntimeError("clean applications must point at real customers")
    if frame["requested_amount"].le(0).any():
        raise RuntimeError("clean requested amounts must be positive")
    if set(frame["product"].unique()) - set(PRODUCTS):
        raise RuntimeError("clean product values must be personal loan or credit card")
    bad_tenure = (frame["product"] == "personal loan") & ~frame["tenure_months"].isin([12, 24, 36, 48, 60])
    if bad_tenure.any():
        raise RuntimeError("personal loans must use a standard tenure")
    if not (frame.loc[frame["product"] == "credit card", "tenure_months"] == 0).all():
        raise RuntimeError("credit cards must have tenure 0")
