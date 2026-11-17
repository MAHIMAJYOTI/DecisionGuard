"""Build the five raw extracts and then damage them.

One random generator is created from the seed and passed through every step.
Do not re-seed inside a step: that would make the files depend on call order
in a way that is harder to explain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_gen.applications import generate_applications
from src.data_gen.bureau import generate_bureau
from src.data_gen.config import DataGenConfig
from src.data_gen.customers import generate_customers
from src.data_gen.performance import generate_performance
from src.data_gen.quality import QualityIssue, inject_quality_issues
from src.data_gen.transactions import generate_transactions

NOTES = (
    "Amounts are Indian rupees. Dates are stored as text because a raw extract is not trusted until a later step parses it.",
    "dpd90_within_12m is blank when the loan is under 12 months old and has not already reached 90 days past due. That blank means the outcome is not known yet. It is not a data-quality error.",
    "The booking rules in performance.py only decide which synthetic loans have a repayment outcome. They are not the credit strategy. Strategies will be versioned YAML files in a later phase.",
    "The bureau file is built from customer attributes and noise. The default flag is created afterwards. The outcome is not copied back into the score.",
    "About 0.5% of clean transactions are fraud. Some of that fraud is ordinary-looking on purpose, so a rulebook cannot catch every row. A few legitimate grocery bursts are included so a velocity rule will also have false positives.",
    "Defects (nulls, duplicate ids, bad dates, negative amounts, orphan keys) are added after the clean checks. The manifest lists exactly what was added.",
)


@dataclass
class GeneratedDataset:
    """Clean-run statistics plus the damaged frames that get written to disk."""

    seed: int
    tables: dict[str, pd.DataFrame]
    issues: list[QualityIssue]
    clean_summary: dict[str, object]
    notes: tuple[str, ...]


def generate_dataset(cfg: DataGenConfig | None = None) -> GeneratedDataset:
    """Create the portfolio extracts. Pass a smaller DataGenConfig for a practice run."""

    cfg = cfg or DataGenConfig()
    cfg.validate()
    rng = np.random.default_rng(cfg.seed)

    customers = generate_customers(rng, cfg)
    bureau = generate_bureau(rng, customers, cfg)
    applications = generate_applications(rng, customers, cfg)
    performance = generate_performance(rng, applications, customers, bureau, cfg)
    transactions, txn_diagnostics = generate_transactions(rng, customers, cfg)

    clean = {
        "customers": customers,
        "credit_bureau": bureau,
        "loan_applications": applications,
        "transactions": transactions,
        "loan_performance": performance,
    }
    summary = _clean_summary(clean, txn_diagnostics, cfg)
    damaged, issues = inject_quality_issues(clean, rng, cfg)
    return GeneratedDataset(
        seed=cfg.seed,
        tables=damaged,
        issues=issues,
        clean_summary=summary,
        notes=NOTES,
    )


def _clean_summary(
    clean: dict[str, pd.DataFrame],
    txn_diagnostics: dict[str, object],
    cfg: DataGenConfig,
) -> dict[str, object]:
    """Statistics of the file BEFORE defects are injected.

    These are the true simulation results. Duplicate rows and blank fraud
    flags change the raw file slightly; do not quote the raw file as the
    fraud rate the generator was asked to build.
    """

    performance = clean["loan_performance"]
    applications = clean["loan_applications"]
    transactions = clean["transactions"]
    full_window = performance["months_on_book"] >= 12
    window_flags = performance.loc[full_window, "dpd90_within_12m"]
    n_window = int(full_window.sum())
    n_bad_window = int((window_flags == 1).sum())
    n_early_bad = int(((performance["months_on_book"] < 12) & (performance["dpd90_within_12m"] == 1)).sum())
    n_fraud = int(transactions["is_fraud"].sum())
    n_txns = len(transactions)
    n_apps = len(applications)
    n_booked = len(performance)
    patterns = txn_diagnostics["fraud_patterns"]
    if not isinstance(patterns, dict):
        raise RuntimeError("fraud pattern diagnostics were not a dict")

    return {
        "seed": cfg.seed,
        "as_of": cfg.as_of,
        "customers": len(clean["customers"]),
        "bureau_rows": len(clean["credit_bureau"]),
        "applications": n_apps,
        "transactions": n_txns,
        "fraud_rows": n_fraud,
        "fraud_rate": n_fraud / n_txns,
        "fraud_patterns": {str(name): int(count) for name, count in patterns.items()},
        "legit_busy_hour_txns": int(txn_diagnostics["legit_busy_hour_txns"]),
        "booked_loans": n_booked,
        "booked_rate": n_booked / n_apps if n_apps else None,
        "loans_with_12m_history": n_window,
        "dpd90_in_12m_window": n_bad_window,
        "dpd90_rate_12m_window": (n_bad_window / n_window) if n_window else None,
        "early_dpd90_before_12m": n_early_bad,
    }
