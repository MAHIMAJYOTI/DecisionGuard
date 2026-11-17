"""Lock every strategy boundary to a written decision.

The score is fixed, so a retrain of the live model does not move these cases.
A real policy change has to update cases.json and list the difference in
expected_changes.json. An unlisted difference fails.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.decision_engine.engine import decide_one
from src.decision_engine.strategy import load_strategy
from tests.test_decision_engine import FixedScore

GOLDEN = Path(__file__).parent / "golden"


def _load(name: str):
    return json.loads((GOLDEN / name).read_text(encoding="utf-8"))


def test_golden_file_has_about_thirty_cases() -> None:
    cases = _load("cases.json")
    assert 30 <= len(cases) <= 40
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))


def test_every_boundary_matches_and_every_change_is_listed() -> None:
    cases = _load("cases.json")
    listed = _load("expected_changes.json")["changes"]
    expected_keys = {
        (row["case_id"], row["from_strategy"], row["to_strategy"]) for row in listed
    }
    seen: set[tuple[str, str, str]] = set()

    for case in cases:
        outputs = {}
        for version, expected in case["expect"].items():
            strategy = load_strategy(version)
            model = FixedScore(case["model_pd"], version=strategy.model_version)
            got = decide_one(case["record"], strategy, model)
            assert got["decision"] == expected["decision"], case["id"]
            assert got["reason_codes"] == expected["reason_codes"], case["id"]
            assert got["strategy_version"] == version
            outputs[version] = got

        versions = list(case["expect"])
        assert len(versions) == 2
        left, right = versions
        differ = (
            outputs[left]["decision"] != outputs[right]["decision"]
            or outputs[left]["reason_codes"] != outputs[right]["reason_codes"]
        )
        key = (case["id"], left, right)
        if differ:
            assert key in expected_keys, f"{case['id']} changed and is not in expected_changes.json"
            seen.add(key)
        else:
            assert key not in expected_keys, f"{case['id']} is listed as a change but the two versions agree"

    missing = expected_keys - seen
    assert not missing, f"expected_changes.json lists differences that did not happen: {missing}"


@pytest.mark.parametrize("case_id", ["credit_pd_016", "credit_thin_pd_009"])
def test_decline_reason_uses_the_renamed_code(case_id: str) -> None:
    cases = {case["id"]: case for case in _load("cases.json")}
    reasons = cases[case_id]["expect"]["credit_v2"]["reason_codes"]
    assert "MODEL_PD_ABOVE_DECLINE_CUTOFF" in reasons
    assert "SCORE_ABOVE_DECLINE_BAND" not in reasons
