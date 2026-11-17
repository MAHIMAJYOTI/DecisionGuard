"""Punch realistic defects into an otherwise clean extract.

The clean generator refuses to emit a null, a duplicate key, a bad date, or a
negative amount. This module adds those afterwards. Staging in Phase 2 is
what should remove them.

Duplicates are appended last, so a copied row can also carry an earlier
defect. That is what a replayed file looks like.

`dpd90_within_12m` is never blanked here. Blanks already mean "the 12-month
outcome is not known yet".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_gen.config import (
    APPLICATION_BAD_CHANNEL,
    APPLICATION_BAD_DATES,
    APPLICATION_BAD_PRODUCT,
    APPLICATION_DUPLICATE_ROWS,
    APPLICATION_NEGATIVE_AMOUNT,
    APPLICATION_NULL_AMOUNT,
    APPLICATION_ORPHAN_CUSTOMER,
    BAD_DATE_TOKENS,
    BUREAU_BAD_UTILISATION,
    BUREAU_DROPPED_ROWS,
    BUREAU_DUPLICATE_ROWS,
    BUREAU_NULL_SCORE,
    BUREAU_SCORE_OUT_OF_RANGE,
    CUSTOMER_BAD_AGE,
    CUSTOMER_DUPLICATE_ROWS,
    CUSTOMER_NEGATIVE_DEBT,
    CUSTOMER_NULL_CITY,
    CUSTOMER_NULL_EMPLOYMENT,
    CUSTOMER_NULL_INCOME,
    DEFAULT_N_APPLICATIONS,
    DEFAULT_N_CUSTOMERS,
    DEFAULT_N_TRANSACTIONS,
    ORPHAN_APPLICATION_ID,
    ORPHAN_CUSTOMER_ID,
    PERFORMANCE_BAD_DATES,
    PERFORMANCE_DUPLICATE_ROWS,
    PERFORMANCE_NEGATIVE_AMOUNT,
    PERFORMANCE_ORPHAN_APPLICATION,
    TRANSACTION_BAD_DATES,
    TRANSACTION_BAD_MERCHANT,
    TRANSACTION_BLANK_COUNTRY,
    TRANSACTION_DUPLICATE_ROWS,
    TRANSACTION_NEGATIVE_AMOUNT,
    TRANSACTION_NULL_AMOUNT,
    TRANSACTION_NULL_FRAUD_FLAG,
    TRANSACTION_ORPHAN_CUSTOMER,
    DataGenConfig,
)
from src.data_gen.utils import scaled_defect_count


@dataclass(frozen=True)
class QualityIssue:
    """One kind of defect written into one file."""

    table: str
    issue: str
    rows_affected: int
    detail: str


class _ColumnEdits:
    """Hands out row positions so two defects do not overwrite the same cell."""

    def __init__(self, n_rows: int, rng: np.random.Generator) -> None:
        self._n = n_rows
        self._rng = rng
        self._used: dict[str, set[int]] = {}

    def take(self, column: str, k: int) -> np.ndarray:
        used = self._used.setdefault(column, set())
        k = min(max(k, 0), self._n - len(used))
        if k <= 0:
            return np.array([], dtype=int)
        picked: list[int] = []
        while len(picked) < k:
            need = k - len(picked)
            draw = self._rng.choice(self._n, size=min(self._n, max(need * 3, need)), replace=False)
            for value in draw:
                position = int(value)
                if position not in used:
                    used.add(position)
                    picked.append(position)
                    if len(picked) == k:
                        break
        return np.array(picked, dtype=int)


def inject_quality_issues(
    tables: dict[str, pd.DataFrame],
    rng: np.random.Generator,
    cfg: DataGenConfig,
) -> tuple[dict[str, pd.DataFrame], list[QualityIssue]]:
    """Return new frames. The clean frames passed in are left untouched."""

    issues: list[QualityIssue] = []
    customers, found = _damage_customers(tables["customers"].copy(), rng)
    issues.extend(found)
    bureau, found = _damage_bureau(tables["credit_bureau"].copy(), rng)
    issues.extend(found)
    applications, found = _damage_applications(tables["loan_applications"].copy(), rng)
    issues.extend(found)
    transactions, found = _damage_transactions(tables["transactions"].copy(), rng)
    issues.extend(found)
    performance, found = _damage_performance(tables["loan_performance"].copy(), rng, cfg)
    issues.extend(found)
    damaged = {
        "customers": customers,
        "credit_bureau": bureau,
        "loan_applications": applications,
        "transactions": transactions,
        "loan_performance": performance,
    }
    return damaged, issues


def _n(full_count: int, n_rows: int, full_rows: int) -> int:
    return scaled_defect_count(full_count, n_rows, full_rows)


def _log(issues: list[QualityIssue], table: str, issue: str, rows: int, detail: str) -> None:
    if rows > 0:
        issues.append(QualityIssue(table, issue, int(rows), detail))


def _col(frame: pd.DataFrame, column: str) -> int:
    location = frame.columns.get_loc(column)
    if isinstance(location, int):
        return location
    raise RuntimeError(f"column {column} is not a single column")


def _put_nulls(frame: pd.DataFrame, column: str, positions: np.ndarray) -> None:
    if len(positions) == 0:
        return
    if pd.api.types.is_integer_dtype(frame[column]):
        frame[column] = frame[column].astype("Int64")
    frame.iloc[positions, _col(frame, column)] = pd.NA


def _put(frame: pd.DataFrame, column: str, positions: np.ndarray, values: object) -> None:
    if len(positions) == 0:
        return
    frame.iloc[positions, _col(frame, column)] = values


def _tokens(options: tuple[str, ...] | list[str], n: int) -> np.ndarray:
    values = np.array(list(options), dtype=object)
    return values[np.arange(n) % len(values)]


def _with_duplicates(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    k: int,
    issues: list[QualityIssue],
    table: str,
    issue: str,
) -> pd.DataFrame:
    k = min(k, len(frame))
    if k <= 0:
        return frame
    copied = frame.iloc[rng.choice(len(frame), size=k, replace=False)].copy()
    updated = pd.concat([frame, copied], ignore_index=True)
    _log(
        issues,
        table,
        issue,
        k,
        "Exact copy of an existing row, including the same id. Appended at the end, so the copy can also contain an earlier defect.",
    )
    return updated


def _damage_customers(
    frame: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, list[QualityIssue]]:
    issues: list[QualityIssue] = []
    edits = _ColumnEdits(len(frame), rng)
    rows = len(frame)

    at = edits.take("income", _n(CUSTOMER_NULL_INCOME, rows, DEFAULT_N_CUSTOMERS))
    _put_nulls(frame, "income", at)
    _log(issues, "customers", "null_income", len(at), "Income left blank.")

    at = edits.take("city", _n(CUSTOMER_NULL_CITY, rows, DEFAULT_N_CUSTOMERS))
    _put_nulls(frame, "city", at)
    _log(issues, "customers", "null_city", len(at), "City left blank.")

    at = edits.take("employment_type", _n(CUSTOMER_NULL_EMPLOYMENT, rows, DEFAULT_N_CUSTOMERS))
    _put_nulls(frame, "employment_type", at)
    _log(issues, "customers", "null_employment_type", len(at), "Employment left blank.")

    at = edits.take("age", _n(CUSTOMER_BAD_AGE, rows, DEFAULT_N_CUSTOMERS))
    if len(at):
        _put(frame, "age", at, np.where(np.arange(len(at)) % 2 == 0, -1, 200))
    _log(
        issues,
        "customers",
        "impossible_age",
        len(at),
        "Age set to -1 or 200. Ages 18-20 are left alone: those customers are valid, a policy may still decline them.",
    )

    at = edits.take("existing_debt", _n(CUSTOMER_NEGATIVE_DEBT, rows, DEFAULT_N_CUSTOMERS))
    if len(at):
        _put(frame, "existing_debt", at, np.round(rng.uniform(-50_000, -1, len(at)), 2))
    _log(issues, "customers", "negative_existing_debt", len(at), "Existing debt set negative.")

    frame = _with_duplicates(
        frame, rng, _n(CUSTOMER_DUPLICATE_ROWS, rows, DEFAULT_N_CUSTOMERS),
        issues, "customers", "duplicate_customer_id",
    )
    return frame, issues


def _damage_bureau(
    frame: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, list[QualityIssue]]:
    issues: list[QualityIssue] = []
    original_rows = len(frame)
    n_drop = _n(BUREAU_DROPPED_ROWS, original_rows, DEFAULT_N_CUSTOMERS)
    drop_at = rng.choice(original_rows, size=n_drop, replace=False)
    frame = frame.drop(frame.index[drop_at]).reset_index(drop=True)
    _log(
        issues,
        "credit_bureau",
        "missing_customer",
        n_drop,
        "Bureau row removed, so the customer has no score to join to.",
    )

    edits = _ColumnEdits(len(frame), rng)
    at = edits.take("bureau_score", _n(BUREAU_NULL_SCORE, original_rows, DEFAULT_N_CUSTOMERS))
    _put_nulls(frame, "bureau_score", at)
    _log(issues, "credit_bureau", "null_bureau_score", len(at), "Bureau score left blank.")

    at = edits.take("bureau_score", _n(BUREAU_SCORE_OUT_OF_RANGE, original_rows, DEFAULT_N_CUSTOMERS))
    if len(at):
        _put(frame, "bureau_score", at, np.where(np.arange(len(at)) % 2 == 0, 120, 1200))
    _log(
        issues,
        "credit_bureau",
        "bureau_score_out_of_range",
        len(at),
        "Score set to 120 or 1200. Valid scores are 300 to 900.",
    )

    at = edits.take("utilisation", _n(BUREAU_BAD_UTILISATION, original_rows, DEFAULT_N_CUSTOMERS))
    if len(at):
        half = max(len(at) // 2, 1) if len(at) > 1 else 0
        if half:
            _put(frame, "utilisation", at[:half], -0.25)
        _put(frame, "utilisation", at[half:], 4.5)
    _log(
        issues,
        "credit_bureau",
        "broken_utilisation",
        len(at),
        "Utilisation set negative or to 4.5. A value such as 1.10 (slightly over limit) is left in the clean rows because that really happens.",
    )

    frame = _with_duplicates(
        frame, rng, _n(BUREAU_DUPLICATE_ROWS, original_rows, DEFAULT_N_CUSTOMERS),
        issues, "credit_bureau", "duplicate_customer_id",
    )
    return frame, issues


def _damage_applications(
    frame: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, list[QualityIssue]]:
    issues: list[QualityIssue] = []
    edits = _ColumnEdits(len(frame), rng)
    rows = len(frame)

    at = edits.take("requested_amount", _n(APPLICATION_NULL_AMOUNT, rows, DEFAULT_N_APPLICATIONS))
    _put_nulls(frame, "requested_amount", at)
    _log(issues, "loan_applications", "null_requested_amount", len(at), "Requested amount left blank.")

    at = edits.take("requested_amount", _n(APPLICATION_NEGATIVE_AMOUNT, rows, DEFAULT_N_APPLICATIONS))
    if len(at):
        _put(frame, "requested_amount", at, np.round(rng.uniform(-100_000, -1, len(at)), 2))
    _log(issues, "loan_applications", "negative_requested_amount", len(at), "Requested amount set negative.")

    at = edits.take("application_ts", _n(APPLICATION_BAD_DATES, rows, DEFAULT_N_APPLICATIONS))
    _put(frame, "application_ts", at, _tokens(BAD_DATE_TOKENS, len(at)))
    _log(
        issues,
        "loan_applications",
        "bad_application_ts",
        len(at),
        "Date replaced with an impossible value, the word not-a-date, a 1900 sentinel, or a date in 2099.",
    )

    at = edits.take("product", _n(APPLICATION_BAD_PRODUCT, rows, DEFAULT_N_APPLICATIONS))
    _put(frame, "product", at, _tokens(("personal_loan", "CREDIT CARD", "", "loan"), len(at)))
    _log(
        issues,
        "loan_applications",
        "bad_product",
        len(at),
        "Product is not exactly 'personal loan' or 'credit card'. Spacing and case matter.",
    )

    at = edits.take("channel", _n(APPLICATION_BAD_CHANNEL, rows, DEFAULT_N_APPLICATIONS))
    _put(frame, "channel", at, _tokens(("phone", "partner", ""), len(at)))
    _log(issues, "loan_applications", "bad_channel", len(at), "Channel is not branch, web, or app.")

    at = edits.take("customer_id", _n(APPLICATION_ORPHAN_CUSTOMER, rows, DEFAULT_N_APPLICATIONS))
    _put(frame, "customer_id", at, ORPHAN_CUSTOMER_ID)
    _log(
        issues,
        "loan_applications",
        "orphan_customer_id",
        len(at),
        "customer_id CUST0000000 does not exist in the customer file.",
    )

    frame = _with_duplicates(
        frame, rng, _n(APPLICATION_DUPLICATE_ROWS, rows, DEFAULT_N_APPLICATIONS),
        issues, "loan_applications", "duplicate_application_id",
    )
    return frame, issues


def _damage_transactions(
    frame: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, list[QualityIssue]]:
    issues: list[QualityIssue] = []
    edits = _ColumnEdits(len(frame), rng)
    rows = len(frame)

    at = edits.take("amount", _n(TRANSACTION_NULL_AMOUNT, rows, DEFAULT_N_TRANSACTIONS))
    _put_nulls(frame, "amount", at)
    _log(issues, "transactions", "null_amount", len(at), "Transaction amount left blank.")

    at = edits.take("amount", _n(TRANSACTION_NEGATIVE_AMOUNT, rows, DEFAULT_N_TRANSACTIONS))
    if len(at):
        _put(frame, "amount", at, np.round(rng.uniform(-5_000, -1, len(at)), 2))
    _log(issues, "transactions", "negative_amount", len(at), "Transaction amount set negative.")

    at = edits.take("txn_ts", _n(TRANSACTION_BAD_DATES, rows, DEFAULT_N_TRANSACTIONS))
    _put(frame, "txn_ts", at, _tokens(BAD_DATE_TOKENS, len(at)))
    _log(issues, "transactions", "bad_txn_ts", len(at), "Transaction timestamp replaced with a value that is not a real date.")

    at = edits.take("country", _n(TRANSACTION_BLANK_COUNTRY, rows, DEFAULT_N_TRANSACTIONS))
    _put(frame, "country", at, "")
    _log(issues, "transactions", "blank_country", len(at), "Country set to an empty string.")

    at = edits.take("merchant_category", _n(TRANSACTION_BAD_MERCHANT, rows, DEFAULT_N_TRANSACTIONS))
    _put(frame, "merchant_category", at, _tokens(("unknown", "NULL", "merch"), len(at)))
    _log(issues, "transactions", "bad_merchant_category", len(at), "Merchant category is not one of the known categories.")

    at = edits.take("is_fraud", _n(TRANSACTION_NULL_FRAUD_FLAG, rows, DEFAULT_N_TRANSACTIONS))
    _put_nulls(frame, "is_fraud", at)
    _log(issues, "transactions", "null_is_fraud", len(at), "Fraud flag left blank. Blank is not the same thing as 'not fraud'.")

    at = edits.take("customer_id", _n(TRANSACTION_ORPHAN_CUSTOMER, rows, DEFAULT_N_TRANSACTIONS))
    _put(frame, "customer_id", at, ORPHAN_CUSTOMER_ID)
    _log(issues, "transactions", "orphan_customer_id", len(at), "customer_id CUST0000000 does not exist in the customer file.")

    frame = _with_duplicates(
        frame, rng, _n(TRANSACTION_DUPLICATE_ROWS, rows, DEFAULT_N_TRANSACTIONS),
        issues, "transactions", "duplicate_txn_id",
    )
    return frame, issues


def _damage_performance(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    cfg: DataGenConfig,
) -> tuple[pd.DataFrame, list[QualityIssue]]:
    """Scale defects from the application count, not from the booked-loan count.

    The booked file is smaller than the application file, but the defect
    budget in config.py is written for a 50,000-application run.
    """

    issues: list[QualityIssue] = []
    if frame.empty:
        return frame, issues

    def planned(full_count: int) -> int:
        count = scaled_defect_count(full_count, cfg.n_applications, DEFAULT_N_APPLICATIONS)
        return min(count, max(len(frame) // 5, 1))

    edits = _ColumnEdits(len(frame), rng)

    at = edits.take("disbursement_ts", planned(PERFORMANCE_BAD_DATES))
    _put(frame, "disbursement_ts", at, _tokens(BAD_DATE_TOKENS, len(at)))
    _log(issues, "loan_performance", "bad_disbursement_ts", len(at), "Disbursement timestamp replaced with a value that is not a real date.")

    at = edits.take("disbursed_amount", planned(PERFORMANCE_NEGATIVE_AMOUNT))
    if len(at):
        _put(frame, "disbursed_amount", at, np.round(rng.uniform(-20_000, -1, len(at)), 2))
    _log(issues, "loan_performance", "negative_disbursed_amount", len(at), "Disbursed amount set negative.")

    at = edits.take("application_id", planned(PERFORMANCE_ORPHAN_APPLICATION))
    _put(frame, "application_id", at, ORPHAN_APPLICATION_ID)
    _log(
        issues,
        "loan_performance",
        "orphan_application_id",
        len(at),
        "application_id APP00000000 does not exist in the application file.",
    )

    frame = _with_duplicates(
        frame, rng, planned(PERFORMANCE_DUPLICATE_ROWS),
        issues, "loan_performance", "duplicate_application_id",
    )
    return frame, issues
