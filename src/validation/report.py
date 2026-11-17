"""Write the strategy-change report from numbers the compare scripts produced."""

from __future__ import annotations

import html
from pathlib import Path

from src.data_gen.config import project_root
from src.models.store import load_metadata

DECISIONS = ("APPROVE", "REFER", "DECLINE")


def write_report(credit: dict, fraud: dict, golden: dict) -> tuple[Path, Path]:
    recommendation = recommend(credit, fraud, golden)
    body = _markdown(credit, fraud, golden, recommendation)
    folder = project_root() / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    markdown_path = folder / "strategy_change_v1_to_v2.md"
    html_path = folder / "strategy_change_v1_to_v2.html"
    markdown_path.write_text(body, encoding="utf-8")
    html_path.write_text(_html(body), encoding="utf-8")
    return markdown_path, html_path


def recommend(credit: dict, fraud: dict, golden: dict) -> dict:
    """Go / no-go from the measured swap and catch rates. Not a fixed sentence."""

    v1 = credit["versions"]["credit_v1"]
    v2 = credit["versions"]["credit_v2"]
    swap = credit["swap_ins"]
    reasons: list[str] = []
    credit_decision = "GO"
    observed_n = swap["n_with_historical_outcome"]
    observed = swap["observed_bad_rate"]
    champion = v1["observed_bad_rate_of_approvals"]
    if observed_n < 50:
        credit_decision = "NO-GO"
        reasons.append(
            f"Credit v2 is NO-GO: only {observed_n} swap-ins have a historical good/bad label, "
            "which is below 50, so the observed bad rate is not evidence."
        )
    elif observed is not None and champion not in (None, 0) and observed > champion * 1.25:
        credit_decision = "NO-GO"
        reasons.append(
            f"Credit v2 is NO-GO: observed swap-in bad rate {observed:.4f} is more than "
            f"1.25 times the champion approval observed bad rate {champion:.4f}."
        )
    else:
        reasons.append(
            f"Credit v2 passes the observed-bad-rate check: swap-in observed bad rate "
            f"{_pct(observed)} on {observed_n:,} labelled rows, champion approvals "
            f"{_pct(champion)} on {v1['n_approvals_with_historical_outcome']:,} labelled rows."
        )
    if v1["expected_loss"]:
        loss_change = v2["expected_loss"] / v1["expected_loss"] - 1.0
        worse_observed = (
            observed is not None and champion is not None and observed > champion
        )
        if loss_change > 0.10 and worse_observed:
            credit_decision = "NO-GO"
            reasons.append(
                f"Credit v2 is NO-GO: expected loss rises {loss_change:.1%} and the "
                "observed swap-in bad rate is higher than the champion approval bad rate."
            )
        else:
            reasons.append(
                f"Expected loss moves {loss_change:+.1%} "
                f"({v1['expected_loss']:,.0f} to {v2['expected_loss']:,.0f} rupees). "
                "LGD is the 0.45 planning assumption, not an observed loss."
            )

    f1 = fraud["versions"]["fraud_v1"]
    f2 = fraud["versions"]["fraud_v2"]
    caught_drop = f1["fraud_caught_rate"] - f2["fraud_caught_rate"]
    if caught_drop > 0.02:
        fraud_decision = "NO-GO"
        reasons.append(
            f"Fraud v2 is NO-GO: fraud caught falls by {caught_drop:.4f}, more than 2 percentage points."
        )
    elif f2["genuine_transactions_blocked"] < f1["genuine_transactions_blocked"]:
        fraud_decision = "CONDITIONAL GO"
        reasons.append(
            f"Fraud v2 is CONDITIONAL GO: fraud caught changes by {-caught_drop:+.4f}, "
            f"inside 2 percentage points, and genuine blocks fall from "
            f"{f1['genuine_transactions_blocked']:,} to {f2['genuine_transactions_blocked']:,}. "
            "The condition is that a CHALLENGE on a velocity burst is an acceptable customer step."
        )
    else:
        fraud_decision = "GO"
        reasons.append("Fraud v2 does not lose more than 2 percentage points of fraud caught.")

    if credit_decision == "NO-GO" or fraud_decision == "NO-GO":
        overall = "NO-GO"
    elif fraud_decision == "CONDITIONAL GO":
        overall = "CONDITIONAL GO"
    else:
        overall = "GO"
    if not golden.get("passed"):
        overall = "NO-GO"
        reasons.append("Golden tests failed, so neither challenger is ready to ship.")
    return {"credit": credit_decision, "fraud": fraud_decision, "overall": overall, "reasons": reasons}


def _markdown(credit: dict, fraud: dict, golden: dict, recommendation: dict) -> str:
    model = _model_section()
    lines = [
        "# Strategy change: v1 to v2",
        "",
        "## Summary",
        "",
        "Credit v2 loosens the bureau cutoff from 600 to 580 and the DTI cutoff from 50% to 55%, "
        "and tightens the thin-file score bands. Fraud v2 turns a 1-hour velocity burst into a "
        "challenge instead of a block, and raises the model challenge cutoff from 0.35 to 0.40. "
        "Both challengers use the same saved models as the champions.",
        "",
        f"Recommendation: **{recommendation['overall']}**.",
        "",
    ]
    for reason in recommendation["reasons"]:
        lines.append(f"- {reason}")
    lines.extend(["", model, "", "## Credit swap-set matrix", "", _matrix(credit), ""])
    lines.extend(["## Credit impact", "", _impact(credit), ""])
    lines.extend(["## Swap-ins and swap-outs", "", _swaps(credit), ""])
    lines.extend(["## Why each swap happened", "", _causes(credit), ""])
    lines.extend(["## Fraud comparison", "", _fraud_table(fraud), "", _fraud_shift(fraud), ""])
    lines.extend(["## Risks", "", _risks(credit), ""])
    lines.extend(["## Golden tests", "", _golden(golden), ""])
    lines.extend(["## Sign-off", "", _signoff(recommendation), ""])
    return "\n".join(lines)


def _model_section() -> str:
    try:
        meta = load_metadata("credit_lr_v1")
    except FileNotFoundError:
        return "Credit model metadata was not found. Train with `python -m src.models`."
    old = meta["previous_unmatured_split"]
    lines = [
        "## Credit model, old split versus matured split",
        "",
        f"Maturity cutoff: application date on or before {meta['maturity_cutoff']} "
        f"(as-of {meta['as_of_date']} minus 12 months). "
        f"Known-outcome rows {meta['n_known_outcome']:,}. Matured rows {meta['n_matured']:,}.",
        "",
        "| | old split | matured split (saved model) |",
        "| --- | ---: | ---: |",
        f"| train bad rate | {old['bad_rate_train']:.4f} | {meta['bad_rate_train']:.4f} |",
        f"| test bad rate | {old['bad_rate_test']:.4f} | {meta['bad_rate_test']:.4f} |",
        f"| AUC | {old['metrics']['auc']:.4f} | {meta['metrics']['auc']:.4f} |",
        f"| Gini | {old['metrics']['gini']:.4f} | {meta['metrics']['gini']:.4f} |",
        f"| KS | {old['metrics']['ks']:.4f} | {meta['metrics']['ks']:.4f} |",
        f"| train n | {old['n_train']:,} | {meta['n_train']:,} |",
        f"| test n | {old['n_test']:,} | {meta['n_test']:,} |",
        "",
        "The old test bad rate is higher because a loan can be labelled bad from month 3 "
        "and good only after month 12. The newest known rows are mostly early defaults. "
        "The saved model drops every application whose 12-month window is still open, "
        "then splits out-of-time inside what remains.",
        "",
        "### Calibration (predicted PD vs actual bad rate)",
        "",
        "| decile | old predicted | old actual | matured predicted | matured actual |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    old_rows = {row["decile"]: row for row in old["calibration_deciles"]}
    new_rows = {row["decile"]: row for row in meta["calibration_deciles"]}
    for decile in sorted(set(old_rows) | set(new_rows)):
        left = old_rows.get(decile)
        right = new_rows.get(decile)
        lines.append(
            "| {decile} | {op} | {oa} | {np} | {na} |".format(
                decile=decile,
                op=_cal(left, "predicted_pd"),
                oa=_cal(left, "actual_bad_rate"),
                np=_cal(right, "predicted_pd"),
                na=_cal(right, "actual_bad_rate"),
            )
        )
    return "\n".join(lines)


def _matrix(credit: dict) -> str:
    lines = [
        "Rows are credit_v1. Columns are credit_v2.",
        "",
        "| v1 \\ v2 | APPROVE | REFER | DECLINE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in credit["swap_matrix"]:
        lines.append(
            f"| {row['v1_decision']} | {row['APPROVE']:,} | {row['REFER']:,} | {row['DECLINE']:,} |"
        )
    return "\n".join(lines)


def _impact(credit: dict) -> str:
    lines = [
        f"Expected loss uses model PD x LGD {credit['lgd']:.2f} x approved limit. "
        f"{credit['lgd_meaning']}.",
        "",
        "| version | approval rate | refer rate | decline rate | expected bad rate of approvals | observed bad rate of approvals | labelled approvals | expected loss (INR) | refers |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for version, impact in credit["versions"].items():
        lines.append(
            "| {version} | {approve:.2%} | {refer:.2%} | {decline:.2%} | {expected:.4f} | {observed} | {labelled:,} | {loss:,.0f} | {refers:,} |".format(
                version=version,
                approve=impact["approve_rate"],
                refer=impact["refer_rate"],
                decline=impact["decline_rate"],
                expected=impact["expected_bad_rate_of_approvals"],
                observed=_pct(impact["observed_bad_rate_of_approvals"]),
                labelled=impact["n_approvals_with_historical_outcome"],
                loss=impact["expected_loss"],
                refers=impact["refer_count"],
            )
        )
    lines.append("")
    lines.append(f"Referral volume change (v2 minus v1): {credit['referral_change']:+,}.")
    return "\n".join(lines)


def _swaps(credit: dict) -> str:
    lines = [
        "Expected bad rate is the average model PD. Observed bad rate uses only rows "
        "the historical book labelled good or bad.",
        "",
        "| slice | count | average model PD (expected) | expected bad rate on labelled rows | expected bad count (sum of PD) | observed bad rate | labelled rows | observed bads |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for title, stats in (
        ("swap-ins: v1 DECLINE or REFER, v2 APPROVE", credit["swap_ins"]),
        ("swap-outs: v1 APPROVE, v2 not APPROVE", credit["swap_outs"]),
    ):
        lines.append(
            "| {title} | {count:,} | {pd} | {expected} | {expected_count} | {observed} | {n:,} | {bads:,} |".format(
                title=title,
                count=stats["count"],
                pd=_pct(stats["average_model_pd"]),
                expected=_pct(stats["expected_bad_rate_on_observed_rows"]),
                expected_count=_pct(stats["expected_bad_count_on_observed_rows"]),
                observed=_pct(stats["observed_bad_rate"]),
                n=stats["n_with_historical_outcome"],
                bads=stats["observed_bad_count"],
            )
        )
    return "\n".join(lines)


def _causes(credit: dict) -> str:
    lines = [
        "Swap-ins are DECLINE or REFER to APPROVE. Swap-outs are APPROVE to anything else. "
        "The other columns are decision changes that do not cross the approve line.",
        "",
        "| cause | swap-ins | swap-outs | DECLINE to APPROVE | DECLINE to REFER | REFER to APPROVE | REFER to DECLINE | APPROVE to REFER | APPROVE to DECLINE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in credit["swap_causes"]:
        lines.append(
            "| {cause} | {swap_ins:,} | {swap_outs:,} | {da:,} | {dr:,} | {ra:,} | {rd:,} | {ar:,} | {ad:,} |".format(
                cause=row["cause"],
                swap_ins=row["swap_ins"],
                swap_outs=row["swap_outs"],
                da=row["DECLINE_to_APPROVE"],
                dr=row["DECLINE_to_REFER"],
                ra=row["REFER_to_APPROVE"],
                rd=row["REFER_to_DECLINE"],
                ar=row["APPROVE_to_REFER"],
                ad=row["APPROVE_to_DECLINE"],
            )
        )
    band = credit["bureau_580_to_599"]
    lines.extend(
        [
            "",
            f"Bureau scores 580-599: {band['rows']:,} applications, of which "
            f"{band['v1_bureau_declines']:,} are declined by v1 for the bureau rule. "
            f"Decision changes in that group: {band['decision_changes']:,}. "
            f"Their mean model PD is {band['mean_model_pd']:.4f} and the lowest is "
            f"{band['min_model_pd']:.4f}, both above the 0.15 decline cutoff. "
            f"{band['declined_by_model_only_in_v2']:,} of them have no remaining hard-rule "
            "decline in v2 and are declined by the model cutoff alone. "
            "The bureau loosening therefore changes the reason on some declines and "
            "does not create approvals.",
        ]
    )
    return "\n".join(lines)


def _fraud_table(fraud: dict) -> str:
    v1 = fraud["versions"]["fraud_v1"]
    v2 = fraud["versions"]["fraud_v2"]
    rows = [
        ("alert rate", "alert_rate", "rate"),
        ("fraud caught", "fraud_caught_rate", "rate"),
        ("fraud value caught (INR)", "fraud_value_caught", "rupees"),
        ("fraud value blocked (INR)", "fraud_value_blocked", "rupees"),
        ("fraud value challenged (INR)", "fraud_value_challenged", "rupees"),
        ("false positives", "false_positives", "count"),
        ("genuine transactions blocked", "genuine_transactions_blocked", "count"),
        ("genuine transactions challenged", "genuine_transactions_challenged", "count"),
        ("genuine customers blocked", "genuine_customers_blocked", "count"),
        ("genuine customers challenged", "genuine_customers_challenged", "count"),
        ("legitimate grocery-burst transactions", "grocery_burst_transactions", "count"),
        ("grocery bursts blocked", "grocery_burst_blocked", "count"),
        ("grocery bursts challenged", "grocery_burst_challenged", "count"),
        ("grocery bursts allowed", "grocery_burst_allowed", "count"),
        ("genuine grocery with 3+ txns in 1 hour", "genuine_grocery_velocity_rows", "count"),
        ("those velocity rows blocked", "genuine_grocery_velocity_blocked", "count"),
        ("those velocity rows challenged", "genuine_grocery_velocity_challenged", "count"),
    ]
    lines = [
        "| metric | fraud_v1 | fraud_v2 |",
        "| --- | ---: | ---: |",
    ]
    for label, key, kind in rows:
        lines.append(f"| {label} | {_cell(v1[key], kind)} | {_cell(v2[key], kind)} |")
    return "\n".join(lines)


def _fraud_shift(fraud: dict) -> str:
    shift = fraud["fraud_rate_shift"]
    return "\n".join(
        [
            "### Why the fraud rate is higher in the model test window",
            "",
            f"Oldest 80% of transactions: fraud rate {shift['train_fraud_rate']:.4f} "
            f"on {shift['train_rows']:,} rows, ending {shift['train_end']}.",
            f"Newest 20%: fraud rate {shift['test_fraud_rate']:.4f} "
            f"on {shift['test_rows']:,} rows, starting {shift['test_start']}.",
            f"Before {shift['fraud_start_day']}: {shift['frauds_before_fraud_window']:,} frauds "
            f"in {shift['rows_before_fraud_window']:,} rows.",
            f"From {shift['fraud_start_day']} onward: {shift['frauds_in_fraud_window']:,} frauds "
            f"in {shift['rows_in_fraud_window']:,} rows "
            f"(rate {shift['fraud_rate_in_fraud_window']:.4f}).",
            "",
            "Legitimate tickets are spread across all 90 days. Fraud tickets are planted "
            "only from day 40, so a 30-day customer history can exist before the fraud. "
            "The oldest 80% of rows includes that quiet stretch, so its fraud rate is lower. "
            "The newest 20% sits inside the window where the fraud was planted.",
        ]
    )


def _risks(credit: dict) -> str:
    return "\n".join(
        [
            "- Reject inference: the credit model only saw booked loans. A swap-in that the old book never funded has a model PD and no observed outcome.",
            "- Outcome maturity: goods need 12 months. The saved model trains only on applications dated on or before the as-of date minus 12 months. Applications after that date are scored, not used to fit.",
            f"- Loss given default is assumed at {credit['lgd']:.2f}. It is not estimated from these files.",
            "- The labelled swap-ins are DTI moves. The old booking policy declined DTI above 60%, so a DTI between 50% and 55% could already have a repayment outcome. Bureau scores 580-599 do not become approvals.",
            "- A grocery burst is identified as five genuine grocery tickets for one customer inside 20 minutes. That is the planted pattern. It is not a column on the mart.",
            "- Fraud value blocked does not change. The drop in fraud value caught is the same size as the drop in fraud value challenged, which is the model challenge cutoff moving from 0.35 to 0.40. A velocity burst whose model score is already at or above 0.80 stays blocked.",
        ]
    )


def _golden(golden: dict) -> str:
    status = "PASS" if golden.get("passed") else "FAIL"
    return "\n".join(
        [
            f"Result: **{status}**. Exit code {golden.get('exit_code')}.",
            "",
            "```",
            golden.get("summary", "").rstrip(),
            "```",
        ]
    )


def _signoff(recommendation: dict) -> str:
    return "\n".join(
        [
            "| role | status |",
            "| --- | --- |",
            "| Business owner | pending |",
            "| Risk | pending |",
            "| IT | pending |",
            "",
            f"Go / no-go recommendation: **{recommendation['overall']}**.",
            "",
            f"- Credit challenger: {recommendation['credit']}",
            f"- Fraud challenger: {recommendation['fraud']}",
            "",
            "Reasons:",
            "",
            *[f"- {reason}" for reason in recommendation["reasons"]],
        ]
    )


def _html(markdown: str) -> str:
    """A readable page. Tables become tables. Everything else stays as paragraphs."""

    parts = [
        "<!DOCTYPE html>",
        "<html lang='en'><head><meta charset='utf-8'>",
        "<title>Strategy change v1 to v2</title>",
        "<style>",
        "body { font-family: Georgia, serif; max-width: 980px; margin: 2rem auto; line-height: 1.45; }",
        "table { border-collapse: collapse; margin: 1rem 0; }",
        "th, td { border: 1px solid #ccc; padding: 0.35rem 0.6rem; text-align: right; }",
        "th:first-child, td:first-child { text-align: left; }",
        "code, pre { font-family: Consolas, monospace; }",
        "</style></head><body>",
    ]
    in_table = False
    in_code = False
    for raw in markdown.splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            if in_code:
                parts.append("</pre>")
                in_code = False
            else:
                parts.append("<pre>")
                in_code = True
            continue
        if in_code:
            parts.append(html.escape(line))
            continue
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if set(line.replace("|", "").replace("-", "").replace(":", "").strip()) == set():
                continue
            if not in_table:
                parts.append("<table>")
                in_table = True
                tag = "th"
            else:
                tag = "td"
            parts.append("<tr>" + "".join(f"<{tag}>{html.escape(cell)}</{tag}>" for cell in cells) + "</tr>")
            continue
        if in_table:
            parts.append("</table>")
            in_table = False
        if line.startswith("# "):
            parts.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            parts.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            parts.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("- "):
            parts.append(f"<p>{_inline(line[2:])}</p>")
        elif line:
            parts.append(f"<p>{_inline(line)}</p>")
    if in_table:
        parts.append("</table>")
    parts.append("</body></html>")
    return "\n".join(parts)


def _inline(text: str) -> str:
    escaped = html.escape(text)
    while "**" in escaped:
        escaped = escaped.replace("**", "<strong>", 1).replace("**", "</strong>", 1)
    return escaped


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _cal(row: dict | None, key: str) -> str:
    if row is None:
        return "n/a"
    return f"{row[key]:.4f}"


def _cell(value: float | int | None, kind: str) -> str:
    if value is None:
        return "n/a"
    if kind == "rate":
        return f"{float(value):.4f}"
    if kind == "rupees":
        return f"{float(value):,.0f}"
    return f"{int(value):,}"
