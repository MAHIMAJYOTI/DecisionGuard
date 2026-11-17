"""Policy rules, schema rejection, and a decision that does not change between calls.

The model in these tests returns a fixed probability so a failure is the rule,
not a change in a fitted score. Boundary values are the ones in the YAML:
a bureau score of exactly 600 is not below 600, a DTI of exactly 50% is not above 50%.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.data_gen.config import project_root
from src.decision_engine.engine import decide_one
from src.decision_engine.log import write_decisions
from src.decision_engine.strategy import StrategyError, load_strategy, load_strategy_file
from src.models.metrics import auc_score, gini_score, ks_score


class FixedScore:
    """Stand-in for a saved model. Every row gets the same probability."""

    def __init__(self, probability: float, version: str = "fixed") -> None:
        self._probability = probability
        self.version = version

    def predict_pd(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self._probability)


def _application(**overrides: object) -> dict:
    record: dict[str, object] = {
        "application_id": "APP1",
        "product": "personal loan",
        "age": 35,
        "income": 600_000,
        "bureau_score": 720,
        "debt_to_income": 0.30,
        "num_delinquencies_24m": 0,
        "num_enquiries_6m": 1,
        "utilisation": 0.40,
        "total_credit_limit": 200_000,
        "requested_amount": 100_000,
    }
    record.update(overrides)
    return record


def _transaction(**overrides: object) -> dict:
    record: dict[str, object] = {
        "txn_id": "TXN1",
        "amount": 500.0,
        "txn_ts": "2026-09-01 12:00:00",
        "txn_count_1h": 0,
        "txn_amount_1h": 0.0,
        "txn_count_24h": 0,
        "txn_amount_24h": 0.0,
        "avg_amount_30d": 400.0,
        "amount_vs_avg_30d": 1.25,
        "is_new_device": 0,
        "is_foreign_country": 0,
        "is_night": 0,
    }
    record.update(overrides)
    return record


def test_policy_boundaries_match_the_yaml() -> None:
    strategy = load_strategy("credit_v1")
    # 1% default probability sits inside the approve band, so only a hard rule can decline.
    model = FixedScore(0.01)

    assert decide_one(_application(age=20), strategy, model)["decision"] == "DECLINE"
    assert decide_one(_application(age=21), strategy, model)["decision"] == "APPROVE"

    declined = decide_one(_application(bureau_score=599), strategy, model)
    allowed = decide_one(_application(bureau_score=600), strategy, model)
    assert declined["decision"] == "DECLINE"
    assert "BUREAU_SCORE_BELOW_600" in declined["reason_codes"]
    assert allowed["decision"] == "APPROVE"
    assert "BUREAU_SCORE_BELOW_600" not in allowed["reason_codes"]

    at_half = decide_one(_application(debt_to_income=0.50), strategy, model)
    over_half = decide_one(_application(debt_to_income=0.51), strategy, model)
    assert at_half["decision"] == "APPROVE"
    assert over_half["decision"] == "DECLINE"
    assert over_half["reason_codes"] == ["DTI_ABOVE_50"]

    assert decide_one(_application(num_delinquencies_24m=2), strategy, model)["decision"] == "APPROVE"
    three = decide_one(_application(num_delinquencies_24m=3), strategy, model)
    assert three["decision"] == "DECLINE"
    assert three["reason_codes"] == ["DELINQUENCIES_24M_AT_LEAST_3"]

    assert decide_one(_application(income=180_000), strategy, model)["decision"] == "APPROVE"
    low_income = decide_one(_application(income=179_999), strategy, model)
    assert low_income["decision"] == "DECLINE"
    assert low_income["reason_codes"] == ["INCOME_BELOW_1_8_LAKH"]


def test_missing_income_is_refer_even_when_the_score_would_decline() -> None:
    """A bureau score of 400 would be a hard decline if we trusted it.

    Income is missing, so the decision is REFER and not an automated decline.
    """

    strategy = load_strategy("credit_v1")
    decision = decide_one(
        _application(income=None, bureau_score=400),
        strategy,
        FixedScore(0.01),
    )
    assert decision["decision"] == "REFER"
    assert decision["reason_codes"] == ["MISSING_INCOME"]
    assert decision["strategy_version"] == "credit_v1"


def test_thin_file_uses_the_tighter_band() -> None:
    strategy = load_strategy("credit_v1")
    # 6% would be approved on a normal file (cutoff 8%) and referred on a thin file (cutoff 5%).
    normal = decide_one(_application(total_credit_limit=10_000), strategy, FixedScore(0.06))
    thin = decide_one(_application(total_credit_limit=0), strategy, FixedScore(0.06))
    assert normal["decision"] == "APPROVE"
    assert thin["decision"] == "REFER"
    assert thin["reason_codes"][0] == "THIN_FILE"


def test_approved_limit_is_capped_by_income_fraction() -> None:
    strategy = load_strategy("credit_v1")
    # Score 742 is the 'low' band: 30% of 720,000 is 216,000, below the 400,000 asked for.
    decision = decide_one(
        _application(bureau_score=742, income=720_000, requested_amount=400_000),
        strategy,
        FixedScore(0.01),
    )
    assert decision["decision"] == "APPROVE"
    assert decision["credit_limit"] == 216_000


def test_fraud_rules_fire_on_the_boundary_and_not_just_below_it() -> None:
    strategy = load_strategy("fraud_v1")
    quiet = FixedScore(0.01)
    assert decide_one(_transaction(txn_count_1h=2), strategy, quiet)["decision"] == "ALLOW"
    blocked = decide_one(_transaction(txn_count_1h=3), strategy, quiet)
    assert blocked["decision"] == "BLOCK"
    assert blocked["reason_codes"][0] == "VELOCITY_1H"

    assert decide_one(_transaction(is_foreign_country=1, is_new_device=0), strategy, quiet)["decision"] == "ALLOW"
    foreign = decide_one(_transaction(is_foreign_country=1, is_new_device=1), strategy, quiet)
    assert foreign["decision"] == "BLOCK"
    assert "FOREIGN_NEW_DEVICE" in foreign["reason_codes"]

    assert decide_one(_transaction(amount_vs_avg_30d=5.0), strategy, quiet)["decision"] == "ALLOW"
    spike = decide_one(_transaction(amount_vs_avg_30d=5.01), strategy, quiet)
    assert spike["decision"] == "CHALLENGE"
    assert spike["reason_codes"][0] == "AMOUNT_VS_AVG_ABOVE_5"

    assert decide_one(_transaction(is_night=1, amount=19_999), strategy, quiet)["decision"] == "ALLOW"
    night = decide_one(_transaction(is_night=1, amount=20_000), strategy, quiet)
    assert night["decision"] == "CHALLENGE"
    assert night["reason_codes"][0] == "NIGHT_HIGH_VALUE"


def test_fraud_score_bands_use_the_yaml_cutoffs() -> None:
    strategy = load_strategy("fraud_v1")
    challenge = float(strategy.challenge_min_pd)
    block = float(strategy.block_min_pd)
    assert decide_one(_transaction(), strategy, FixedScore(challenge - 0.001))["decision"] == "ALLOW"
    assert decide_one(_transaction(), strategy, FixedScore(challenge))["decision"] == "CHALLENGE"
    assert decide_one(_transaction(), strategy, FixedScore(block))["decision"] == "BLOCK"


def test_invalid_yaml_is_rejected(tmp_path: Path) -> None:
    text = (project_root() / "strategies" / "credit_strategy_v1.yaml").read_text(encoding="utf-8")
    broken = tmp_path / "broken.yaml"
    broken.write_text(text.replace("decision: REFER", "decision: APPROVE", 1), encoding="utf-8")
    with pytest.raises(StrategyError):
        load_strategy_file(broken)


def test_same_input_and_versions_always_match() -> None:
    strategy = load_strategy("credit_v1")
    model = FixedScore(0.04, version="credit_lr_v1")
    record = _application(age=20, debt_to_income=0.62, num_delinquencies_24m=4)
    first = decide_one(record, strategy, model)
    second = decide_one(record, strategy, model)
    assert first == second
    assert first["decision"] == "DECLINE"
    assert first["reason_codes"] == [
        "AGE_BELOW_21",
        "DTI_ABOVE_50",
        "DELINQUENCIES_24M_AT_LEAST_3",
    ]
    assert first["strategy_version"] == "credit_v1"
    assert first["model_version"] == "credit_lr_v1"


def test_challenger_strategy_loads_with_the_looser_cutoffs() -> None:
    strategy = load_strategy("credit_v2")
    cutoffs = {rule.reason: rule.value for rule in strategy.policy_rules}
    assert cutoffs["BUREAU_SCORE_BELOW_580"] == 580
    assert cutoffs["DTI_ABOVE_55"] == 0.55
    assert strategy.thin_file is not None
    assert strategy.thin_file.approve_max_pd < strategy.approve_max_pd


def test_decision_log_appends_a_row(tmp_path: Path) -> None:
    strategy = load_strategy("credit_v1")
    model = FixedScore(0.01, version="credit_lr_v1")
    record = _application()
    decision = decide_one(record, strategy, model)
    db_path = tmp_path / "audit.duckdb"
    write_decisions(db_path, pd.DataFrame([record]), pd.DataFrame([decision]), "runtest")
    connection = duckdb.connect(str(db_path))
    try:
        row = connection.execute(
            "SELECT decision, strategy_version, model_version, reason_codes FROM decisions.decision_log"
        ).fetchone()
    finally:
        connection.close()
    assert row is not None
    assert row[0] == "APPROVE"
    assert row[1] == "credit_v1"
    assert row[2] == "credit_lr_v1"
    assert "SCORE_WITHIN_APPROVE_BAND" in row[3]


def test_ranking_metrics_on_a_perfect_score() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    auc = auc_score(labels, scores)
    assert auc == pytest.approx(1.0)
    assert gini_score(auc) == pytest.approx(1.0)
    assert ks_score(labels, scores) == pytest.approx(1.0)
