"""Loan performance extract. One row per *booked* application.

Declined applications are absent. That is how servicing data works: there is
no repayment history for a loan that was never given. The booking rules in
this file are a historical underwriting policy used only to decide who has
an outcome. They are not the governed strategy (that is Phase 3, and it
lives in YAML so it can be changed without editing this code).

Outcome window
--------------
90 days past due cannot happen in the first three months, so those loans
have a blank `dpd90_within_12m`. Between 3 and 12 months the flag is 1 only
if the bad event has already happened, otherwise blank (still too early to
call it a good loan). At 12 months or more the flag is 1 or 0.

A blank flag here is "not known yet". It is not a data-quality defect.
quality.py does not punch extra blanks into this column, so Phase 2 can
tell the two situations apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_gen.config import (
    ASSUMED_PERSONAL_LOAN_APR,
    BOOK_MAX_DELINQUENCIES,
    BOOK_MAX_DTI,
    BOOK_MIN_AGE,
    BOOK_MIN_INCOME,
    BOOK_MIN_SCORE,
    BOOK_REFER_APPROVE_RATE,
    BOOK_REFER_SCORE,
    BOOK_STRAIGHT_THROUGH_RATE,
    CREDIT_CARD_OBLIGATION_FRACTION,
    EXISTING_DEBT_EMI_FRACTION,
    PERFORMANCE_COLUMNS,
    DataGenConfig,
)


def personal_loan_emi(
    principal: np.ndarray,
    tenure_months: np.ndarray,
    apr: float = ASSUMED_PERSONAL_LOAN_APR,
) -> np.ndarray:
    """Standard annuity payment. Same formula a spreadsheet PMT uses.

    Example: 100,000 INR at 16% for 12 months is about 9,073 INR a month.
    """

    monthly_rate = apr / 12.0
    n_months = tenure_months.astype(float)
    factor = (1.0 + monthly_rate) ** n_months
    return principal * monthly_rate * factor / (factor - 1.0)


def generate_performance(
    rng: np.random.Generator,
    applications: pd.DataFrame,
    customers: pd.DataFrame,
    bureau: pd.DataFrame,
    cfg: DataGenConfig,
) -> pd.DataFrame:
    """Simulate booking, then simulate whether booked loans reach 90+ DPD."""

    base = applications.merge(customers, on="customer_id", how="left", validate="many_to_one")
    base = base.merge(bureau, on="customer_id", how="left", validate="many_to_one")
    if base["bureau_score"].isna().any() or base["income"].isna().any():
        raise RuntimeError("every clean application should match a customer and a bureau row")

    dti = _debt_to_income(base)
    booked = _booked_mask(rng, base, dti)
    if not booked.any():
        return _empty()

    booked_rows = base.loc[booked].reset_index(drop=True)
    outcomes = _outcomes(rng, booked_rows, dti[booked], cfg)
    frame = pd.DataFrame(
        {
            "application_id": booked_rows["application_id"].to_numpy(),
            "customer_id": booked_rows["customer_id"].to_numpy(),
            "disbursed_amount": _disbursed_amount(rng, booked_rows),
            "disbursement_ts": outcomes["disbursement_ts"],
            "months_on_book": outcomes["months_on_book"],
            "max_dpd_12m": outcomes["max_dpd_12m"],
            "dpd90_within_12m": outcomes["dpd90_within_12m"],
        }
    )
    _check(frame, applications)
    return frame[list(PERFORMANCE_COLUMNS)]


def _debt_to_income(base: pd.DataFrame) -> np.ndarray:
    """Monthly obligations divided by monthly income.

    Existing debt is a stock, so it is turned into a rough monthly payment
    with a flat 3% factor (a stand-in for the bureau's obligation file, which
    we do not have). A new personal loan uses a 16% annuity. A new card uses
    5% of the limit, which is a common minimum-due proxy.

    Trade-off: this is good enough to *simulate* who would have been booked.
    Phase 2 should recompute DTI in SQL from the raw columns, not trust a
    number we could have stamped here. We deliberately do not store DTI.
    """

    income = base["income"].to_numpy(dtype=float)
    monthly_income = np.maximum(income / 12.0, 1.0)
    existing_emi = base["existing_debt"].to_numpy(dtype=float) * EXISTING_DEBT_EMI_FRACTION
    requested = base["requested_amount"].to_numpy(dtype=float)
    tenure = base["tenure_months"].to_numpy(dtype=int)
    product = base["product"].to_numpy()

    new_obligation = np.zeros(len(base))
    loan = product == "personal loan"
    card = product == "credit card"
    if loan.any():
        new_obligation[loan] = personal_loan_emi(requested[loan], tenure[loan])
    if card.any():
        new_obligation[card] = requested[card] * CREDIT_CARD_OBLIGATION_FRACTION

    dti = (existing_emi + new_obligation) / monthly_income
    # No income cannot support a new facility. A huge DTI makes the rule decline.
    dti = np.where(income <= 0, 99.0, dti)
    return dti


def _booked_mask(rng: np.random.Generator, base: pd.DataFrame, dti: np.ndarray) -> np.ndarray:
    """Who the historical policy actually booked.

    Hard declines: under 21, score below 580, DTI above 60%, 3+ delinquencies,
    or income below 1.8 lakh. Scores from 580 up to 640 are a refer band and
    are booked only 45% of the time (a manual underwriter). Everyone else is
    booked 97% of the time; the missing 3% stands in for incomplete files,
    which the data does not show directly.
    """

    hard_decline = (
        (base["age"].to_numpy() < BOOK_MIN_AGE)
        | (base["bureau_score"].to_numpy() < BOOK_MIN_SCORE)
        | (dti > BOOK_MAX_DTI)
        | (base["num_delinquencies_24m"].to_numpy() >= BOOK_MAX_DELINQUENCIES)
        | (base["income"].to_numpy() < BOOK_MIN_INCOME)
    )
    refer = (~hard_decline) & (base["bureau_score"].to_numpy() < BOOK_REFER_SCORE)
    threshold = np.where(refer, BOOK_REFER_APPROVE_RATE, BOOK_STRAIGHT_THROUGH_RATE)
    roll = rng.random(len(base))
    return (~hard_decline) & (roll < threshold)


def _disbursed_amount(rng: np.random.Generator, booked_rows: pd.DataFrame) -> np.ndarray:
    """Loans are sometimes cut slightly. Cards are opened at the requested limit."""

    requested = booked_rows["requested_amount"].to_numpy(dtype=float)
    product = booked_rows["product"].to_numpy()
    haircut = rng.uniform(0.92, 1.0, len(booked_rows))
    disbursed = np.where(product == "personal loan", requested * haircut, requested)
    return np.round(disbursed, 2)


def _outcomes(
    rng: np.random.Generator,
    booked_rows: pd.DataFrame,
    dti: np.ndarray,
    cfg: DataGenConfig,
) -> dict[str, np.ndarray | pd.Series]:
    """Draw the 90+ DPD flag from a simple logistic curve of the risk drivers.

    The coefficients are assumptions for the simulator, not a fitted scorecard.
    They only encode the usual pattern: a lower bureau score, a higher DTI,
    prior delinquencies and high utilisation make 90+ DPD more likely.
    """

    n = len(booked_rows)
    applied = pd.to_datetime(booked_rows["application_ts"])
    lag_days = rng.integers(1, 8, n)
    disbursed_at = applied + pd.to_timedelta(lag_days, unit="D")
    as_of = pd.Timestamp(cfg.as_of)
    # A file dated as_of cannot contain a disbursement that has not happened.
    disbursed_at = disbursed_at.where(disbursed_at <= as_of, as_of)
    months = (as_of.year - disbursed_at.dt.year) * 12 + (as_of.month - disbursed_at.dt.month)
    months = months.clip(lower=0).to_numpy(dtype=int)

    probability = _default_probability(booked_rows, dti)
    visible = np.where(months >= 12, probability, np.where(months >= 3, probability * (months / 12.0), 0.0))
    is_bad = rng.random(n) < visible

    flag = np.full(n, np.nan)
    flag = np.where(is_bad, 1.0, flag)
    flag = np.where(~is_bad & (months >= 12), 0.0, flag)

    mild_dpd = rng.choice([0, 0, 0, 0, 0, 7, 15, 29, 35, 59], size=n)
    cap = np.maximum(months, 0) * 30
    max_dpd = np.where(is_bad, rng.integers(90, 181, n), np.minimum(mild_dpd, cap))

    flag_series = pd.Series(flag).astype("Float64").astype("Int64")
    return {
        "disbursement_ts": disbursed_at.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
        "months_on_book": months.astype(int),
        "max_dpd_12m": max_dpd.astype(int),
        "dpd90_within_12m": flag_series,
    }


def _default_probability(booked_rows: pd.DataFrame, dti: np.ndarray) -> np.ndarray:
    """12-month probability of 90+ DPD for a loan that was booked.

    At a score of 720, DTI 0.35, no delinquencies and ordinary utilisation
    this lands near 7%. Weaker booked loans (the old refer band) land higher.
    The intercept was chosen so the *booked* book, not the whole applicant
    pool, sits in a high-single-digit bad rate. Check the manifest after a
    run rather than trusting this comment if you change the coefficients.
    """

    score = booked_rows["bureau_score"].to_numpy(dtype=float)
    delinquencies = booked_rows["num_delinquencies_24m"].to_numpy(dtype=float)
    utilisation = booked_rows["utilisation"].to_numpy(dtype=float)
    enquiries = booked_rows["num_enquiries_6m"].to_numpy(dtype=float)
    employment = booked_rows["employment_type"].to_numpy()
    product = booked_rows["product"].to_numpy()

    logit = (
        -2.55
        - 0.012 * (score - 700.0)
        + 1.55 * (np.minimum(dti, 1.5) - 0.35)
        + 0.42 * delinquencies
        + 0.85 * (utilisation - 0.40)
        + 0.10 * enquiries
        + np.where(employment == "self_employed", 0.22, 0.0)
        + np.where(product == "credit card", 0.12, 0.0)
    )
    logit = np.clip(logit, -8, 5)
    return 1.0 / (1.0 + np.exp(-logit))


def _empty() -> pd.DataFrame:
    frame = pd.DataFrame(columns=list(PERFORMANCE_COLUMNS))
    return frame


def _check(frame: pd.DataFrame, applications: pd.DataFrame) -> None:
    if frame.empty:
        return
    if not set(frame["application_id"]).issubset(set(applications["application_id"])):
        raise RuntimeError("performance rows must point at real applications")
    flag = frame["dpd90_within_12m"]
    known = flag.dropna()
    if not set(known.unique()).issubset({0, 1}):
        raise RuntimeError("dpd90_within_12m must be 0, 1, or blank")
    # eq() on a nullable integer column returns <NA> where the flag is blank.
    is_bad = flag.eq(1).fillna(False).astype(bool)
    if is_bad.any() and frame.loc[is_bad, "max_dpd_12m"].lt(90).any():
        raise RuntimeError("a 90+ flag requires max_dpd_12m of at least 90")
    early = frame["months_on_book"].lt(3) & flag.notna()
    if early.any():
        raise RuntimeError("loans under 3 months cannot have a known 90+ outcome")
