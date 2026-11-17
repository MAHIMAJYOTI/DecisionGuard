"""Customer extract. One row per person.

Columns match the brief: age, income, employment, city, existing debt,
months at the current address. Amounts are Indian rupees (annual income,
outstanding debt). Names and government ids are left out on purpose: a
decisioning example does not need them.

Zero income is allowed for some unemployed customers. That is a real value,
not a data-quality error. Broken values (null income, age 200, negative debt)
are added later by quality.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_gen.config import CUSTOMER_COLUMNS, EMPLOYMENT_TYPES, DataGenConfig
from src.data_gen.utils import customer_ids, round_money

# (city, share of the book). Shares sum to 1.
_CITIES: tuple[tuple[str, float], ...] = (
    ("Mumbai", 0.16),
    ("Delhi", 0.14),
    ("Bengaluru", 0.13),
    ("Hyderabad", 0.09),
    ("Chennai", 0.08),
    ("Pune", 0.07),
    ("Kolkata", 0.06),
    ("Ahmedabad", 0.05),
    ("Jaipur", 0.05),
    ("Lucknow", 0.04),
    ("Kochi", 0.04),
    ("Chandigarh", 0.03),
    ("Indore", 0.03),
    ("Coimbatore", 0.03),
)

# log(income) mean and spread, by employment. The mean is ln(a typical salary),
# so the *median* person lands near that salary and a few people earn much more.
_INCOME_LOG: dict[str, tuple[float, float]] = {
    "salaried": (13.30, 0.40),        # median about 6.0 lakh
    "self_employed": (13.15, 0.65),   # wider spread, same ballpark
    "student": (11.00, 0.45),         # median about 60 thousand
    "retired": (12.61, 0.35),         # median about 3.0 lakh
    "unemployed": (10.82, 0.35),      # median about 50 thousand, before the zeros
    "homemaker": (11.70, 0.50),       # median about 1.2 lakh
}

_INCOME_CAP: dict[str, float] = {
    "salaried": 4_000_000,
    "self_employed": 8_000_000,
    "student": 400_000,
    "retired": 2_000_000,
    "unemployed": 300_000,
    "homemaker": 900_000,
}


def generate_customers(rng: np.random.Generator, cfg: DataGenConfig) -> pd.DataFrame:
    """Build the clean customer file (no injected defects yet)."""

    n = cfg.n_customers
    age = _ages(rng, n)
    employment = _employment(rng, age)
    income = _income(rng, employment)
    debt = _existing_debt(rng, income, employment)
    months = _months_at_address(rng, age)

    frame = pd.DataFrame(
        {
            "customer_id": customer_ids(1, n),
            "age": age.astype(int),
            "income": income,
            "employment_type": employment,
            "city": _cities(rng, n),
            "existing_debt": debt,
            "months_at_address": months.astype(int),
        }
    )
    _check(frame)
    return frame[list(CUSTOMER_COLUMNS)]


def _ages(rng: np.random.Generator, n: int) -> np.ndarray:
    """Adult ages centred near 38, plus a visible group of 18-20 year olds.

    The 18-20 group is valid data. A credit policy may still decline them
    (the usual floor is 21). That decline is a business rule, not a data fix,
    so these ages must stay inside the clean file.
    """

    age = np.clip(np.rint(rng.normal(38, 11, n)), 18, 78).astype(int)
    n_young = int(round(0.035 * n))
    young_at = rng.choice(n, size=n_young, replace=False)
    age[young_at] = rng.integers(18, 21, size=n_young)
    return age


def _employment(rng: np.random.Generator, age: np.ndarray) -> np.ndarray:
    """Employment depends on age, so we don't create 19-year-old retirees."""

    employment = np.empty(len(age), dtype=object)
    young = age < 22
    older = age >= 60
    mid = ~young & ~older

    if young.any():
        employment[young] = rng.choice(
            ["student", "salaried", "unemployed"],
            size=int(young.sum()),
            p=[0.72, 0.18, 0.10],
        )
    if older.any():
        employment[older] = rng.choice(
            ["retired", "salaried", "self_employed"],
            size=int(older.sum()),
            p=[0.78, 0.14, 0.08],
        )
    if mid.any():
        employment[mid] = rng.choice(
            ["salaried", "self_employed", "homemaker", "unemployed", "student", "retired"],
            size=int(mid.sum()),
            p=[0.60, 0.22, 0.07, 0.05, 0.04, 0.02],
        )
    return employment


def _income(rng: np.random.Generator, employment: np.ndarray) -> np.ndarray:
    """Annual income in INR. Most unemployed customers are stored as exact zeros."""

    income = np.zeros(len(employment))
    for name, (mu, sigma) in _INCOME_LOG.items():
        mask = employment == name
        if not mask.any():
            continue
        drawn = rng.lognormal(mu, sigma, int(mask.sum()))
        income[mask] = np.minimum(drawn, _INCOME_CAP[name])

    unemployed = employment == "unemployed"
    no_earnings = unemployed & (rng.random(len(employment)) < 0.70)
    income[no_earnings] = 0.0
    return round_money(income)


def _existing_debt(
    rng: np.random.Generator,
    income: np.ndarray,
    employment: np.ndarray,
) -> np.ndarray:
    """Outstanding debt. About one in five customers has none.

    The amount is a fraction of income (a bit higher for self-employed
    customers, who more often borrow against uneven cash flow). Someone with
    no income can still have a small leftover balance.
    """

    n = len(income)
    ratio = rng.beta(1.6, 4.0, n)
    self_employed_boost = np.where(employment == "self_employed", 1.4, 1.0)
    debt = income * ratio * self_employed_boost
    no_income = income <= 0
    debt[no_income] = rng.uniform(0, 25_000, int(no_income.sum()))
    has_no_debt = rng.random(n) < 0.22
    debt[has_no_debt] = 0.0
    return round_money(debt)


def _months_at_address(rng: np.random.Generator, age: np.ndarray) -> np.ndarray:
    """Cannot have lived somewhere longer than they have been an adult."""

    upper = np.maximum((age - 17) * 12, 1)
    return np.floor(rng.random(len(age)) * (upper + 1)).astype(int)


def _cities(rng: np.random.Generator, n: int) -> np.ndarray:
    names = [name for name, _share in _CITIES]
    shares = [share for _name, share in _CITIES]
    return rng.choice(names, size=n, p=shares)


def _check(frame: pd.DataFrame) -> None:
    """Stop if the clean file already contains a defect. Those belong in quality.py."""

    if frame["customer_id"].duplicated().any():
        raise RuntimeError("clean customer ids must be unique")
    if frame["age"].lt(18).any() or frame["age"].gt(78).any():
        raise RuntimeError("clean ages must sit between 18 and 78")
    if frame["income"].lt(0).any() or frame["existing_debt"].lt(0).any():
        raise RuntimeError("clean money columns cannot be negative")
    unknown = set(frame["employment_type"].unique()) - set(EMPLOYMENT_TYPES)
    if unknown:
        raise RuntimeError(f"unexpected employment values: {unknown}")
