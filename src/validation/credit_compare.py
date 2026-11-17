"""Score the same applications with credit v1 and credit v2.

Swap-ins are rows the champion would not approve and the challenger would.
Swap-outs are the reverse. Expected bad rate is the average model probability.
Observed bad rate uses only rows the historical book actually labelled good or bad.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.data_gen.config import project_root
from src.decision_engine.engine import decide_frame
from src.decision_engine.strategy import load_strategy
from src.models.store import load_model
from src.validation.config import LGD
from src.warehouse.load import default_db_path

DECISIONS = ("APPROVE", "REFER", "DECLINE")


def compare_credit(db_path: Path | None = None) -> dict:
    path = db_path or default_db_path()
    model = load_model("credit_lr_v1")
    champion = load_strategy("credit_v1")
    challenger = load_strategy("credit_v2")
    frame = _load(path)
    left = decide_frame(frame, champion, model)
    right = decide_frame(frame, challenger, model)
    outcome = frame["outcome_status"].astype(str)
    matrix = _matrix(left["decision"], right["decision"])
    swap_in = (left["decision"].isin(["DECLINE", "REFER"])) & (right["decision"] == "APPROVE")
    swap_out = (left["decision"] == "APPROVE") & (right["decision"] != "APPROVE")
    causes = _causes(left, right)
    result = {
        "lgd": LGD,
        "lgd_meaning": "assumed share of the approved limit lost if the loan reaches 90+ DPD",
        "model_version": model.version,
        "n_applications": int(len(frame)),
        "swap_matrix": matrix,
        "swap_ins": _slice_stats(swap_in.to_numpy(), right, outcome, "model PD on swap-ins"),
        "swap_outs": _slice_stats(swap_out.to_numpy(), left, outcome, "model PD on swap-outs"),
        "versions": {
            "credit_v1": _impact(left, outcome),
            "credit_v2": _impact(right, outcome),
        },
        "swap_causes": _cause_table(causes, left["decision"], right["decision"]),
        "bureau_580_to_599": _bureau_band(frame, left, right),
    }
    result["referral_change"] = (
        result["versions"]["credit_v2"]["refer_count"]
        - result["versions"]["credit_v1"]["refer_count"]
    )
    _print(result)
    return result


def _load(path: Path) -> pd.DataFrame:
    connection = duckdb.connect(str(path), read_only=True)
    try:
        frame = connection.execute(
            """
            SELECT
                application_id,
                age,
                income,
                bureau_score,
                debt_to_income,
                num_delinquencies_24m,
                num_enquiries_6m,
                utilisation,
                total_credit_limit,
                requested_amount,
                outcome_status
            FROM marts.mart_credit_applications
            """
        ).fetchdf()
    finally:
        connection.close()
    return frame.reset_index(drop=True)


def _matrix(left: pd.Series, right: pd.Series) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for v1 in DECISIONS:
        row: dict[str, int | str] = {"v1_decision": v1}
        for v2 in DECISIONS:
            row[v2] = int(((left == v1) & (right == v2)).sum())
        rows.append(row)
    return rows


def _slice_stats(mask: np.ndarray, scored: pd.DataFrame, outcome: pd.Series, label: str) -> dict:
    """Expected numbers use the model. Observed numbers use historical labels."""

    count = int(mask.sum())
    scores = scored.loc[mask, "score"].astype(float)
    known = mask & outcome.isin(["good", "bad"]).to_numpy()
    known_scores = scored.loc[known, "score"].astype(float)
    observed_bad = outcome.loc[known].eq("bad")
    return {
        "label": label,
        "count": count,
        "average_model_pd": float(scores.mean()) if count else None,
        "expected_bad_rate": float(scores.mean()) if count else None,
        "expected_bad_rate_on_observed_rows": float(known_scores.mean()) if known.any() else None,
        "observed_bad_rate": float(observed_bad.mean()) if known.any() else None,
        "n_with_historical_outcome": int(known.sum()),
        "observed_bad_count": int(observed_bad.sum()) if known.any() else 0,
        "expected_bad_count_on_observed_rows": float(known_scores.sum()) if known.any() else None,
        "numbers": {
            "average_model_pd": "model-expected",
            "expected_bad_rate": "model-expected (same figure as average PD)",
            "expected_bad_rate_on_observed_rows": "model-expected, restricted to rows with a historical label",
            "observed_bad_rate": "observed on the historical book",
        },
    }


def _impact(decided: pd.DataFrame, outcome: pd.Series) -> dict:
    total = len(decided)
    counts = {name: int((decided["decision"] == name).sum()) for name in DECISIONS}
    approved = (decided["decision"] == "APPROVE").to_numpy()
    scores = decided.loc[approved, "score"].astype(float)
    limits = decided.loc[approved, "credit_limit"].astype(float)
    expected_loss = float((scores.to_numpy() * LGD * limits.to_numpy()).sum()) if approved.any() else 0.0
    known = approved & outcome.isin(["good", "bad"]).to_numpy()
    observed = outcome.loc[known].eq("bad")
    return {
        "approve_count": counts["APPROVE"],
        "refer_count": counts["REFER"],
        "decline_count": counts["DECLINE"],
        "approve_rate": counts["APPROVE"] / total,
        "refer_rate": counts["REFER"] / total,
        "decline_rate": counts["DECLINE"] / total,
        "expected_bad_rate_of_approvals": float(scores.mean()) if approved.any() else None,
        "observed_bad_rate_of_approvals": float(observed.mean()) if known.any() else None,
        "n_approvals_with_historical_outcome": int(known.sum()),
        "expected_loss": expected_loss,
        "expected_loss_note": "sum of model PD x LGD x approved limit, in rupees",
    }


def _causes(left: pd.DataFrame, right: pd.DataFrame) -> pd.Series:
    """One primary cause per row, from the reason codes that actually changed."""

    causes: list[str] = []
    for v1_reasons, v2_reasons in zip(left["reason_codes"], right["reason_codes"], strict=True):
        v1 = set(v1_reasons)
        v2 = set(v2_reasons)
        bureau = "BUREAU_SCORE_BELOW_600" in v1 and "BUREAU_SCORE_BELOW_580" not in v2
        dti = "DTI_ABOVE_50" in v1 and "DTI_ABOVE_55" not in v2
        thin = ("THIN_FILE" in v1 or "THIN_FILE" in v2) and (
            "SCORE_WITHIN_APPROVE_BAND" in v1
            or "SCORE_WITHIN_REFER_BAND" in v1
            or "MODEL_PD_ABOVE_DECLINE_CUTOFF" in v1
            or "SCORE_WITHIN_APPROVE_BAND" in v2
            or "SCORE_WITHIN_REFER_BAND" in v2
            or "MODEL_PD_ABOVE_DECLINE_CUTOFF" in v2
        )
        if bureau and dti:
            causes.append("bureau_cutoff_and_dti_cutoff")
        elif bureau:
            causes.append("bureau_cutoff_600_to_580")
        elif dti:
            causes.append("dti_cutoff_50_to_55")
        elif thin and v1_reasons != v2_reasons:
            causes.append("thin_file_score_band")
        elif v1_reasons != v2_reasons:
            causes.append("other_reason_change")
        else:
            causes.append("unchanged")
    return pd.Series(causes)


_TRANSITIONS = (
    ("APPROVE", "REFER"),
    ("APPROVE", "DECLINE"),
    ("REFER", "APPROVE"),
    ("REFER", "DECLINE"),
    ("DECLINE", "APPROVE"),
    ("DECLINE", "REFER"),
)


def _cause_table(
    causes: pd.Series,
    left: pd.Series,
    right: pd.Series,
) -> list[dict[str, int | str]]:
    """One row per rule change. Columns are the decision moves, not just approvals."""

    rows: list[dict[str, int | str]] = []
    labels = [
        "bureau_cutoff_600_to_580",
        "dti_cutoff_50_to_55",
        "bureau_cutoff_and_dti_cutoff",
        "thin_file_score_band",
        "other_reason_change",
    ]
    for label in labels:
        mask = (causes == label).to_numpy()
        row: dict[str, int | str] = {"cause": label}
        for source, target in _TRANSITIONS:
            moved = mask & (left == source).to_numpy() & (right == target).to_numpy()
            row[f"{source}_to_{target}"] = int(moved.sum())
        row["swap_ins"] = int(row["DECLINE_to_APPROVE"]) + int(row["REFER_to_APPROVE"])
        row["swap_outs"] = int(row["APPROVE_to_REFER"]) + int(row["APPROVE_to_DECLINE"])
        rows.append(row)
    accounted = sum(int(row[f"{source}_to_{target}"]) for row in rows for source, target in _TRANSITIONS)
    changed = int((left != right).sum())
    if accounted != changed:
        raise RuntimeError(
            f"Cause table explains {accounted:,} decision changes but {changed:,} rows changed."
        )
    return rows


def _bureau_band(frame: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame) -> dict:
    """Scores of 580-599 are inside the challenger's new bureau window.

    The model still scores that band far above the 0.15 decline cutoff, so
    loosening the hard rule does not change the decision.
    """

    score = pd.to_numeric(frame["bureau_score"], errors="coerce")
    band = score.between(580, 599)
    had_bureau_reason = left["reason_codes"].map(lambda reasons: "BUREAU_SCORE_BELOW_600" in reasons)
    cohort = band & had_bureau_reason
    model_pd = right.loc[cohort, "score"].astype(float)
    model_only = right.loc[cohort, "reason_codes"].map(
        lambda reasons: reasons == ["MODEL_PD_ABOVE_DECLINE_CUTOFF"]
    )
    return {
        "rows": int(band.sum()),
        "v1_bureau_declines": int(cohort.sum()),
        "decision_changes": int((cohort & left["decision"].ne(right["decision"])).sum()),
        "mean_model_pd": float(model_pd.mean()) if len(model_pd) else None,
        "min_model_pd": float(model_pd.min()) if len(model_pd) else None,
        "declined_by_model_only_in_v2": int(model_only.sum()),
    }


def _print(result: dict) -> None:
    print("Credit swap-set matrix (rows v1, columns v2)")
    header = f"{'v1 / v2':<12}" + "".join(f"{name:>10}" for name in DECISIONS)
    print(header)
    for row in result["swap_matrix"]:
        line = f"{row['v1_decision']:<12}"
        for name in DECISIONS:
            line += f"{row[name]:>10,}"
        print(line)
    print()
    _print_slice("Swap-ins (v1 DECLINE or REFER -> v2 APPROVE)", result["swap_ins"])
    _print_slice("Swap-outs (v1 APPROVE -> v2 not APPROVE)", result["swap_outs"])
    print("Impact (expected = model PD, observed = historical good/bad labels)")
    print(
        f"  {'version':<12} {'approve':>10} {'refer':>10} {'decline':>10} "
        f"{'E[bad|appr]':>12} {'obs bad|appr':>14} {'E[loss] INR':>16} {'refers':>10}"
    )
    for version, impact in result["versions"].items():
        observed = impact["observed_bad_rate_of_approvals"]
        observed_text = f"{observed:.4f}" if observed is not None else "n/a"
        print(
            f"  {version:<12} {impact['approve_rate']:>10.2%} {impact['refer_rate']:>10.2%} "
            f"{impact['decline_rate']:>10.2%} {impact['expected_bad_rate_of_approvals']:>12.4f} "
            f"{observed_text:>14} {impact['expected_loss']:>16,.0f} {impact['refer_count']:>10,}"
        )
    print(f"  Referral volume change (v2 - v1): {result['referral_change']:+,}")
    print("Swap causes")
    print(
        f"  {'cause':<32} {'in':>6} {'out':>6} "
        f"{'D->A':>6} {'D->R':>6} {'R->A':>6} {'R->D':>6} {'A->R':>6} {'A->D':>6}"
    )
    for row in result["swap_causes"]:
        print(
            f"  {row['cause']:<32} {row['swap_ins']:>6,} {row['swap_outs']:>6,} "
            f"{row['DECLINE_to_APPROVE']:>6,} {row['DECLINE_to_REFER']:>6,} "
            f"{row['REFER_to_APPROVE']:>6,} {row['REFER_to_DECLINE']:>6,} "
            f"{row['APPROVE_to_REFER']:>6,} {row['APPROVE_to_DECLINE']:>6,}"
        )
    band = result["bureau_580_to_599"]
    print(
        "Bureau 580-599: "
        f"{band['rows']:,} applications, {band['v1_bureau_declines']:,} declined by v1 for bureau, "
        f"{band['decision_changes']:,} decision changes. "
        f"Mean model PD {band['mean_model_pd']:.4f}, minimum {band['min_model_pd']:.4f}. "
        f"{band['declined_by_model_only_in_v2']:,} are declined by v2 for the model cutoff alone."
    )


def _print_slice(title: str, stats: dict) -> None:
    print(title)
    print(f"  count: {stats['count']:,}")
    print(f"  average model PD (model-expected): {_fmt(stats['average_model_pd'])}")
    print(f"  expected bad rate (model-expected): {_fmt(stats['expected_bad_rate'])}")
    print(
        "  expected bad rate on rows with a historical outcome (model-expected): "
        f"{_fmt(stats['expected_bad_rate_on_observed_rows'])}"
    )
    print(
        f"  observed bad rate (historical): {_fmt(stats['observed_bad_rate'])} "
        f"on {stats['n_with_historical_outcome']:,} labelled rows "
        f"({stats['observed_bad_count']:,} bad)"
    )
    print(
        "  expected bad count on those labelled rows (sum of model PD): "
        f"{_fmt(stats['expected_bad_count_on_observed_rows'])}"
    )


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def main() -> None:
    result = compare_credit()
    out = project_root() / "reports" / "credit_compare.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
