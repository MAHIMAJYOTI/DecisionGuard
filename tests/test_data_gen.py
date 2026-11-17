"""Tests for the Phase 1 generator.

The full 500,000-row file is produced by `python -m src.data_gen`.
These tests use a smaller config so they finish in a few seconds, and they
check the contracts the full run also has to keep.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_gen.config import (
    APPLICATION_COLUMNS,
    BUREAU_COLUMNS,
    CUSTOMER_COLUMNS,
    PERFORMANCE_COLUMNS,
    TRANSACTION_COLUMNS,
    DataGenConfig,
)
from src.data_gen.dataset import generate_dataset
from src.data_gen.io import write_raw_files
from src.data_gen.performance import personal_loan_emi
from src.data_gen.utils import scaled_defect_count, split_budget

# Large enough that fraud patterns and a 12-month outcome both appear,
# small enough to rerun inside a test.
PRACTICE = DataGenConfig(n_customers=1_500, n_applications=1_500, n_transactions=12_000)


def test_annuity_matches_closed_form() -> None:
    """100,000 INR at 16% for 12 months. The number is the standard EMI formula."""

    emi = personal_loan_emi(np.array([100_000.0]), np.array([12]), apr=0.16)
    monthly = 0.16 / 12.0
    expected = 100_000.0 * monthly * (1.0 + monthly) ** 12 / ((1.0 + monthly) ** 12 - 1.0)
    assert emi[0] == pytest.approx(expected)
    assert emi[0] == pytest.approx(9073.0, abs=1.0)


def test_split_budget_uses_every_row() -> None:
    shares = {"a": 0.28, "b": 0.24, "c": 0.18, "d": 0.15, "e": 0.15}
    assert sum(split_budget(2500, shares).values()) == 2500
    assert sum(split_budget(20, shares).values()) == 20


def test_full_size_defect_count_is_exact() -> None:
    assert scaled_defect_count(750, 50_000, 50_000) == 750
    assert scaled_defect_count(1_000, 500_000, 500_000) == 1_000


def test_same_seed_rebuilds_the_same_extracts() -> None:
    first = generate_dataset(PRACTICE)
    second = generate_dataset(PRACTICE)
    for name in first.tables:
        pd.testing.assert_frame_equal(first.tables[name], second.tables[name])


def test_a_different_seed_changes_income() -> None:
    first = generate_dataset(PRACTICE)
    other = generate_dataset(DataGenConfig(seed=99, n_customers=1_500, n_applications=1_500, n_transactions=12_000))
    assert not first.tables["customers"]["income"].equals(other.tables["customers"]["income"])


def test_clean_summary_matches_the_brief() -> None:
    dataset = generate_dataset(PRACTICE)
    summary = dataset.clean_summary
    assert summary["customers"] == PRACTICE.n_customers
    assert summary["applications"] == PRACTICE.n_applications
    assert summary["transactions"] == PRACTICE.n_transactions
    assert summary["fraud_rows"] == PRACTICE.n_fraud
    assert summary["fraud_rate"] == pytest.approx(0.005)
    assert summary["booked_loans"] > 0
    assert summary["loans_with_12m_history"] > 0
    patterns = summary["fraud_patterns"]
    assert isinstance(patterns, dict)
    assert sum(patterns.values()) == PRACTICE.n_fraud


def test_raw_columns_match_the_schemas() -> None:
    dataset = generate_dataset(PRACTICE)
    assert tuple(dataset.tables["customers"].columns) == CUSTOMER_COLUMNS
    assert tuple(dataset.tables["credit_bureau"].columns) == BUREAU_COLUMNS
    assert tuple(dataset.tables["loan_applications"].columns) == APPLICATION_COLUMNS
    assert tuple(dataset.tables["transactions"].columns) == TRANSACTION_COLUMNS
    assert tuple(dataset.tables["loan_performance"].columns) == PERFORMANCE_COLUMNS


def test_injected_defects_are_actually_in_the_frames() -> None:
    dataset = generate_dataset(PRACTICE)
    customers = dataset.tables["customers"]
    bureau = dataset.tables["credit_bureau"]
    applications = dataset.tables["loan_applications"]
    transactions = dataset.tables["transactions"]
    performance = dataset.tables["loan_performance"]

    assert customers["income"].isna().any()
    assert customers["customer_id"].duplicated().any()
    assert customers["age"].isin([-1, 200]).any()
    assert (customers["existing_debt"].dropna() < 0).any()

    scores = bureau["bureau_score"].dropna()
    assert ((scores < 300) | (scores > 900)).any()
    assert bureau["customer_id"].duplicated().any()

    assert applications["application_ts"].isin(["2024-13-45", "not-a-date"]).any()
    assert (applications["requested_amount"].dropna() < 0).any()
    assert (applications["customer_id"] == "CUST0000000").any()
    assert applications["application_id"].duplicated().any()

    assert (transactions["amount"].dropna() < 0).any()
    assert transactions["amount"].isna().any()
    assert transactions["txn_id"].duplicated().any()
    assert transactions["is_fraud"].isna().any()

    assert (performance["application_id"] == "APP00000000").any()
    flags = set(performance["dpd90_within_12m"].dropna().unique())
    assert flags.issubset({0, 1})


def test_known_performance_rows_still_point_at_real_applications() -> None:
    """Orphan performance keys are injected on purpose. Every other key must match."""

    dataset = generate_dataset(PRACTICE)
    performance = dataset.tables["loan_performance"]
    applications = dataset.tables["loan_applications"]
    real = performance["application_id"] != "APP00000000"
    assert performance.loc[real, "application_id"].isin(applications["application_id"]).all()


def test_writer_creates_csv_and_parquet(tmp_path: object) -> None:
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    dataset = generate_dataset(DataGenConfig(seed=7, n_customers=200, n_applications=200, n_transactions=2_000))
    manifest = write_raw_files(dataset, tmp_path)
    assert manifest.exists()
    reloaded = pd.read_parquet(tmp_path / "customers.parquet")
    assert len(reloaded) == len(dataset.tables["customers"])
    assert (tmp_path / "transactions.csv").exists()
