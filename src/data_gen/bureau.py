"""Credit bureau extract. One row per customer.

Built only from the customer attributes (and noise). The loan outcome is
created later, from this file. We never write the outcome back into the
score, so the features do not contain the answer.

Trade-off: a production bureau table is a history of pulls, and you join the
latest pull at or before the application timestamp. The brief asks for one
row per customer, so that point-in-time join is not simulated. Do not treat
this snapshot as if it were a feature store.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_gen.config import BUREAU_COLUMNS, DataGenConfig
from src.data_gen.utils import round_money, sigmoid

_EMPLOYMENT_RISK: dict[str, float] = {
    "salaried": -0.35,
    "self_employed": 0.25,
    "homemaker": 0.15,
    "retired": 0.05,
    "student": 0.70,
    "unemployed": 1.10,
}


def generate_bureau(
    rng: np.random.Generator,
    customers: pd.DataFrame,
    cfg: DataGenConfig,
) -> pd.DataFrame:
    """One bureau row for every clean customer."""

    risk = _risk_index(customers)
    score = _bureau_score(rng, risk)
    enquiries = rng.poisson(0.30 + 1.80 * sigmoid(risk))
    # A high risk index raises the chance of prior delinquencies. The lambda
    # stays small for the average customer, so "3 or more" is uncommon but real.
    delinquency_lambda = 0.05 + 0.70 * sigmoid(risk)
    delinquency_lambda = np.where(risk > 1.6, delinquency_lambda + 0.90, delinquency_lambda)
    delinquencies = rng.poisson(delinquency_lambda)
    credit_limit = _credit_limit(rng, customers, score)
    utilisation = _utilisation(rng, risk, credit_limit)

    frame = pd.DataFrame(
        {
            "customer_id": customers["customer_id"].to_numpy(),
            "bureau_score": score.astype(int),
            "num_enquiries_6m": enquiries.astype(int),
            "num_delinquencies_24m": delinquencies.astype(int),
            "total_credit_limit": credit_limit,
            "utilisation": utilisation,
        }
    )
    _check(frame, customers, cfg)
    return frame[list(BUREAU_COLUMNS)]


def _risk_index(customers: pd.DataFrame) -> np.ndarray:
    """A unit-free risk number. Higher means the bureau should look worse.

    It is standardised to mean 0 and standard deviation 1 so the score formula
    below is easy to read: each +1 of risk pulls the score down by a fixed amount.
    This column is NOT written to the file. If it were, the model would be
    cheating by reading our recipe.
    """

    income = customers["income"].to_numpy(dtype=float)
    debt = customers["existing_debt"].to_numpy(dtype=float)
    age = customers["age"].to_numpy(dtype=int)
    months = customers["months_at_address"].to_numpy(dtype=int)
    employment = customers["employment_type"].to_numpy()

    # np.where would still divide by zero on the incomes we are about to replace.
    debt_ratio = np.full(len(income), 3.0)
    has_income = income > 0
    debt_ratio[has_income] = debt[has_income] / income[has_income]
    debt_ratio = np.clip(debt_ratio, 0, 6)
    employment_risk = np.array([_EMPLOYMENT_RISK[value] for value in employment])

    raw = (
        employment_risk
        + np.where(age < 25, 0.55, 0.0)
        + np.where(age > 65, 0.25, 0.0)
        + 0.85 * (debt_ratio - 0.30)
        + np.where(months < 6, 0.45, 0.0)
    )
    standard = raw.std()
    if standard == 0:
        return np.zeros(len(raw))
    return (raw - raw.mean()) / standard


def _bureau_score(rng: np.random.Generator, risk: np.ndarray) -> np.ndarray:
    """Score from 300 to 900. 700 is roughly the middle of this book.

    Noise of +/- 35 points stops the score being a pure formula of income,
    which would make the later model unrealistically perfect.
    """

    noise = rng.normal(0, 35, len(risk))
    score = np.rint(705 - 62 * risk + noise)
    return np.clip(score, 300, 900)


def _credit_limit(
    rng: np.random.Generator,
    customers: pd.DataFrame,
    score: np.ndarray,
) -> np.ndarray:
    """Total limit across the customer's existing facilities. Thin files get 0.

    About 8% have no credit file limit yet (new to credit). Everyone else is
    scaled by income and by score, so a high score tends to come with a higher
    limit. That correlation is what lenders see in bureau data.
    """

    income = customers["income"].to_numpy(dtype=float)
    thin_file = rng.random(len(income)) < 0.08
    base = np.maximum(income, 80_000) * rng.uniform(0.15, 1.60, len(income))
    score_scale = 0.35 + 0.65 * (score / 900.0)
    limit = np.where(thin_file, 0.0, base * score_scale)
    return round_money(limit)


def _utilisation(
    rng: np.random.Generator,
    risk: np.ndarray,
    credit_limit: np.ndarray,
) -> np.ndarray:
    """Balance divided by limit. 0.40 means 40% of the limit is drawn.

    A value a little above 1 is kept: customers do go over limit, and a later
    data test should not treat 1.10 as corruption. Negative utilisation is
    the corruption, and quality.py adds that separately. No limit means no
    utilisation (stored as 0, not null).
    """

    mean_u = np.clip(0.18 + 0.55 * sigmoid(risk), 0.08, 0.85)
    concentration = 8.0
    alpha = np.maximum(mean_u * concentration, 0.4)
    beta = np.maximum((1.0 - mean_u) * concentration, 0.4)
    utilisation = rng.beta(alpha, beta)
    over_limit = rng.random(len(risk)) < 0.02
    utilisation = np.where(over_limit, utilisation + rng.uniform(0.05, 0.18, len(risk)), utilisation)
    utilisation = np.clip(utilisation, 0, 1.20)
    utilisation = np.where(credit_limit <= 0, 0.0, utilisation)
    return np.round(utilisation, 4)


def _check(frame: pd.DataFrame, customers: pd.DataFrame, cfg: DataGenConfig) -> None:
    if len(frame) != cfg.n_customers:
        raise RuntimeError("clean bureau file must have one row per customer")
    if not frame["customer_id"].is_unique:
        raise RuntimeError("clean bureau file must be one row per customer")
    if set(frame["customer_id"]) != set(customers["customer_id"]):
        raise RuntimeError("clean bureau file must cover every customer exactly once")
    score = frame["bureau_score"]
    if score.lt(300).any() or score.gt(900).any():
        raise RuntimeError("clean bureau scores must stay inside 300-900")
    if frame["utilisation"].lt(0).any() or frame["utilisation"].gt(1.20).any():
        raise RuntimeError("clean utilisation must stay inside 0 to 1.20")
    if frame["num_delinquencies_24m"].lt(0).any() or frame["num_enquiries_6m"].lt(0).any():
        raise RuntimeError("clean enquiry and delinquency counts cannot be negative")
