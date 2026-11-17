"""DecisionGuard demo. Reads the snapshot in app/data. No warehouse.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP = Path(__file__).resolve().parent
DATA = APP / "data"
TEAL = "#0E6B6B"
INK = "#1B2430"
AMBER = "#A15C07"
RED = "#9B2335"
GREEN = "#1F7A4D"

st.set_page_config(
    page_title="DecisionGuard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; max-width: 1180px; }
    .dg-kicker { letter-spacing: 0.14em; text-transform: uppercase;
                 font-size: 0.75rem; color: #0E6B6B; font-weight: 700; margin: 0; }
    .dg-title { font-size: 2rem; line-height: 1.15; margin: 0.15rem 0 0.3rem; color: #1B2430; }
    .dg-lead { color: #3d4a5c; font-size: 1.05rem; margin-bottom: 0.6rem; }
    .flow { display: flex; flex-wrap: wrap; gap: 0.45rem; align-items: center; margin: 0.6rem 0 1rem; }
    .step { background: white; border: 1px solid #ddd4c8; border-radius: 8px;
            padding: 0.45rem 0.7rem; font-size: 0.92rem; color: #1B2430; }
    .arrow { color: #0E6B6B; font-weight: 700; }
    .verdict { border-radius: 10px; padding: 0.9rem 1rem; margin: 0.4rem 0 0.8rem; }
    .verdict h3 { margin: 0 0 0.35rem; }
    .verdict.NO-GO { background: #f8e8eb; border: 1px solid #9B2335; }
    .verdict.GO { background: #e5f4ec; border: 1px solid #1F7A4D; }
    .verdict.CONDITIONAL { background: #f8f1e3; border: 1px solid #A15C07; }
    .note { background: #fff; border-left: 4px solid #0E6B6B; padding: 0.7rem 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_json(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


@st.cache_data
def load_portfolio() -> pd.DataFrame:
    return pd.read_parquet(DATA / "portfolio_summary.parquet")


@st.cache_resource
def load_runtime():
    """Models and strategies. Loaded once, then reused."""

    import joblib

    from src.decision_engine.strategy import load_strategy_file

    models = {
        version: joblib.load(APP / "models" / version / "model.joblib")
        for version in ("credit_lr_v1", "fraud_hgb_v1")
    }
    strategies = {
        path.stem: load_strategy_file(path)
        for path in sorted((APP / "strategies").glob("*.yaml"))
    }
    by_version = {strategy.version: strategy for strategy in strategies.values()}
    return models, by_version


def main() -> None:
    if not (DATA / "manifest.json").exists():
        st.error("The demo snapshot is missing. From the project folder run: python -m src.demo.export")
        st.stop()
    st.markdown('<p class="dg-kicker">DecisionGuard</p>', unsafe_allow_html=True)
    st.markdown('<p class="dg-title">Credit and fraud decisions, on a synthetic book</p>', unsafe_allow_html=True)
    overview, quality, credit, fraud, change, try_it = st.tabs(
        [
            "Overview",
            "Data quality",
            "Credit decisions",
            "Fraud",
            "Strategy change",
            "Try it",
        ]
    )
    with overview:
        _overview()
    with quality:
        _quality()
    with credit:
        _credit()
    with fraud:
        _fraud()
    with change:
        _change()
    with try_it:
        _try_it()


def _overview() -> None:
    kpis = load_json("portfolio_kpis.json")
    manifest = load_json("manifest.json")
    samples = load_json("samples.json")
    st.markdown(
        '<p class="dg-lead">DecisionGuard takes a loan application or a card transaction, '
        "applies a written strategy, scores it with a saved model, and keeps the reason codes. "
        "Every number on this page was exported from the local warehouse. "
        "The people and the payments are synthetic.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="flow">
          <div class="step">Raw files</div><div class="arrow">→</div>
          <div class="step">Staging</div><div class="arrow">→</div>
          <div class="step">Marts</div><div class="arrow">→</div>
          <div class="step">Decision engine</div><div class="arrow">→</div>
          <div class="step">Decision log</div><div class="arrow">→</div>
          <div class="step">Validation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    columns = st.columns(4)
    columns[0].metric("Applications", f"{kpis['applications']:,}")
    columns[1].metric("Booked", f"{kpis['booked']:,}")
    columns[2].metric("Card transactions", f"{kpis['transactions']:,}")
    columns[3].metric("Fraud rate", f"{kpis['fraud_rate']:.2%}")
    columns = st.columns(4)
    columns[0].metric("Customers", f"{kpis['customers']:,}")
    columns[1].metric("Good loans", f"{kpis['good_loans']:,}")
    columns[2].metric("Bad loans", f"{kpis['bad_loans']:,}")
    columns[3].metric("Disbursed", _rupees(kpis["disbursed_amount"]))
    monthly = load_portfolio()
    monthly = monthly[monthly["period_grain"] == "month"].dropna(subset=["application_count"])
    monthly = monthly.sort_values("period_start")
    figure = go.Figure()
    figure.add_bar(x=monthly["period_start"], y=monthly["application_count"], name="Applications", marker_color=TEAL)
    figure.add_bar(x=monthly["period_start"], y=monthly["booked_count"], name="Booked", marker_color="#8FB8B8")
    figure.update_layout(barmode="group", title="Applications and bookings by month")
    _style(figure)
    st.plotly_chart(figure, use_container_width=True)
    st.markdown(
        f'<div class="note">Snapshot exported {manifest["exported_at"]}. '
        f"Models {', '.join(manifest['models'])}. "
        f"The demo carries {samples['credit_rows']:,} application rows and "
        f"{samples['fraud_rows']:,} transaction rows "
        f"({samples['fraud_rows_in_sample']:,} of them fraud), plus the summaries above. "
        f"app/data is {manifest['data_bytes'] / 1024 / 1024:.2f} MB. "
        "DuckDB and dbt are not used when this app is running.</div>",
        unsafe_allow_html=True,
    )


def _quality() -> None:
    quality = load_json("data_quality.json")
    tables = pd.DataFrame(quality["tables"])
    tables["staging_plus_rejected"] = tables["staging_rows"] + tables["rejected_rows"]
    st.markdown(
        "Raw is what landed. Staging is what passed the checks. "
        "Rejected is the count of distinct raw rows that failed at least one check. "
        "Staging plus rejected equals raw."
    )
    st.dataframe(
        tables.rename(
            columns={
                "table_name": "Table",
                "raw_rows": "Raw",
                "staging_rows": "Staging",
                "rejected_rows": "Rejected rows",
                "staging_plus_rejected": "Staging + rejected",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    reasons = pd.DataFrame(quality["reasons"])
    figure = go.Figure(
        go.Bar(
            y=reasons["table_name"] + " · " + reasons["reject_reason"],
            x=reasons["rejected_rows"],
            orientation="h",
            marker_color=TEAL,
        )
    )
    figure.update_layout(title="Rejects by reason", yaxis_title="", xaxis_title="Reason rows")
    _style(figure, height=520)
    st.plotly_chart(figure, use_container_width=True)
    st.caption(quality["rejected_meaning"])


def _credit() -> None:
    mix = load_json("decision_mix.json")
    model = load_json("credit_model.json")
    decisions = pd.DataFrame(mix["decisions"])
    left, right = st.columns([1, 1])
    with left:
        figure = go.Figure(
            go.Bar(
                x=decisions["decision"],
                y=decisions["count"],
                marker_color=[GREEN, AMBER, RED],
                text=[f"{rate:.1%}" for rate in decisions["rate"]],
                textposition="outside",
            )
        )
        figure.update_layout(title=f"Decision mix · {mix['strategy_version']}", yaxis_title="Applications")
        _style(figure)
        st.plotly_chart(figure, use_container_width=True)
    with right:
        reasons = pd.DataFrame(mix["top_decline_reasons"])
        figure = go.Figure(
            go.Bar(x=reasons["count"], y=reasons["reason"], orientation="h", marker_color=RED)
        )
        figure.update_layout(title="Top decline reasons", yaxis={"categoryorder": "total ascending"})
        _style(figure)
        st.plotly_chart(figure, use_container_width=True)
    st.caption(mix["decline_reason_note"])
    metrics = model["metrics"]
    columns = st.columns(3)
    columns[0].metric("AUC", f"{metrics['auc']:.4f}")
    columns[1].metric("Gini", f"{metrics['gini']:.4f}")
    columns[2].metric("KS", f"{metrics['ks']:.4f}")
    st.caption(
        f"Matured out-of-time test. Train bad rate {model['bad_rate_train']:.4f} "
        f"on {model['n_train']:,} loans. Test bad rate {model['bad_rate_test']:.4f} "
        f"on {model['n_test']:,} loans. Applications on or before {model['maturity_cutoff']}."
    )
    calibration = pd.DataFrame(model["calibration_deciles"])
    figure = go.Figure()
    figure.add_scatter(
        x=calibration["decile"], y=calibration["predicted_pd"], name="Predicted PD", mode="lines+markers", line_color=TEAL
    )
    figure.add_scatter(
        x=calibration["decile"], y=calibration["actual_bad_rate"], name="Actual bad rate", mode="lines+markers", line_color=RED
    )
    figure.update_layout(title="Calibration by decile (1 = lowest predicted PD)", xaxis_title="Decile")
    _style(figure)
    st.plotly_chart(figure, use_container_width=True)


def _fraud() -> None:
    model = load_json("fraud_model.json")
    table = pd.DataFrame(model["threshold_table"])
    metrics = model["metrics"]
    columns = st.columns(4)
    columns[0].metric("AUC", f"{metrics['auc']:.4f}")
    columns[1].metric("PR-AUC", f"{metrics['pr_auc']:.4f}")
    columns[2].metric("Test fraud rate", f"{model['fraud_rate_test']:.2%}")
    columns[3].metric("Test rows", f"{model['n_test']:,}")
    st.caption(
        "The threshold table is the out-of-time test. "
        f"Train fraud rate was {model['fraud_rate_train']:.2%} because fraud is planted only from day 40."
    )
    display = table.copy()
    display["threshold"] = display["threshold"].map(lambda value: f"{value:.2f}")
    display["alert_rate"] = display["alert_rate"].map(lambda value: f"{value:.2%}")
    display["precision"] = display["precision"].map(lambda value: f"{value:.2%}")
    display["fraud_caught"] = display["fraud_caught"].map(lambda value: f"{value:.2%}")
    display["false_positives"] = display["false_positives"].map(lambda value: f"{int(value):,}")
    st.dataframe(
        display.rename(
            columns={
                "threshold": "Threshold",
                "alert_rate": "Alert rate",
                "precision": "Precision",
                "fraud_caught": "Fraud caught",
                "false_positives": "False positives",
                "alerts": "Alerts",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    figure = go.Figure()
    figure.add_scatter(x=table["threshold"], y=table["alert_rate"], name="Alert rate", mode="lines+markers", line_color=TEAL)
    figure.add_scatter(x=table["threshold"], y=table["fraud_caught"], name="Fraud caught", mode="lines+markers", line_color=GREEN)
    figure.add_scatter(
        x=table["threshold"],
        y=table["false_positives"] / model["n_test"],
        name="False positives / test rows",
        mode="lines+markers",
        line_color=AMBER,
    )
    figure.update_layout(title="Alert rate, fraud caught, and false positives", xaxis_title="Score threshold")
    _style(figure)
    st.plotly_chart(figure, use_container_width=True)
    averages = pd.DataFrame(model["feature_averages"])
    genuine = averages[averages["is_fraud"] == 0].iloc[0]
    fraud = averages[averages["is_fraud"] == 1].iloc[0]
    features = [column for column in averages.columns if column not in {"is_fraud", "transactions"}]
    compare = pd.DataFrame(
        {
            "Feature": features,
            "Genuine": [float(genuine[feature]) for feature in features],
            "Fraud": [float(fraud[feature]) for feature in features],
        }
    )
    st.markdown(
        f"Feature averages on the full mart. "
        f"Genuine transactions: {int(genuine['transactions']):,}. "
        f"Fraud transactions: {int(fraud['transactions']):,}."
    )
    st.dataframe(compare, hide_index=True, use_container_width=True)
    flags = ["is_new_device", "is_foreign_country", "is_night", "amount_vs_avg_30d"]
    figure = go.Figure()
    figure.add_bar(name="Genuine", x=flags, y=[float(genuine[name]) for name in flags], marker_color="#8FB8B8")
    figure.add_bar(name="Fraud", x=flags, y=[float(fraud[name]) for name in flags], marker_color=RED)
    figure.update_layout(barmode="group", title="Flags and amount vs 30-day average")
    _style(figure)
    st.plotly_chart(figure, use_container_width=True)


def _change() -> None:
    payload = load_json("strategy_change.json")
    golden = load_json("golden.json")
    recommendation = payload["recommendation"]
    verdict = recommendation["overall"].split()[0]
    css_class = "CONDITIONAL" if verdict.startswith("CONDITIONAL") else verdict
    reasons = "".join(f"<li>{reason}</li>" for reason in recommendation["reasons"])
    st.markdown(
        f'<div class="verdict {css_class}"><h3>{recommendation["overall"]}</h3>'
        f"<p>Credit challenger: {recommendation['credit']}. "
        f"Fraud challenger: {recommendation['fraud']}.</p><ul>{reasons}</ul></div>",
        unsafe_allow_html=True,
    )
    credit = payload["credit"]
    matrix = pd.DataFrame(credit["swap_matrix"]).set_index("v1_decision")
    matrix = matrix[["APPROVE", "REFER", "DECLINE"]]
    figure = go.Figure(
        go.Heatmap(
            z=matrix.values,
            x=list(matrix.columns),
            y=list(matrix.index),
            colorscale=[[0, "#F4F1EC"], [1, TEAL]],
            text=[[f"{int(value):,}" for value in row] for row in matrix.values],
            texttemplate="%{text}",
            colorbar_title="Applications",
        )
    )
    figure.update_layout(
        title="Swap-set matrix · rows v1, columns v2",
        xaxis_title="credit_v2",
        yaxis_title="credit_v1",
        yaxis_autorange="reversed",
    )
    _style(figure, height=420)
    st.plotly_chart(figure, use_container_width=True)
    impact = []
    for version, row in credit["versions"].items():
        impact.append(
            {
                "Version": version,
                "Approval": f"{row['approve_rate']:.2%}",
                "Refer": f"{row['refer_rate']:.2%}",
                "Decline": f"{row['decline_rate']:.2%}",
                "Expected bad rate of approvals": f"{row['expected_bad_rate_of_approvals']:.4f}",
                "Observed bad rate of approvals": f"{row['observed_bad_rate_of_approvals']:.4f}",
                "Expected loss (INR)": f"{row['expected_loss']:,.0f}",
                "Refers": f"{row['refer_count']:,}",
            }
        )
    st.markdown(
        f"Expected loss is model PD × {credit['lgd']:.2f} × approved limit. {credit['lgd_meaning']}."
    )
    st.dataframe(pd.DataFrame(impact), hide_index=True, use_container_width=True)
    st.caption(f"Referral volume change, v2 minus v1: {credit['referral_change']:+,}.")
    swaps = []
    for label, stats in (("Swap-ins", credit["swap_ins"]), ("Swap-outs", credit["swap_outs"])):
        swaps.append(
            {
                "Slice": label,
                "Count": f"{stats['count']:,}",
                "Average model PD": f"{stats['average_model_pd']:.4f}",
                "Expected bad count on labelled rows": f"{stats['expected_bad_count_on_observed_rows']:.2f}",
                "Observed bad rate": f"{stats['observed_bad_rate']:.4f}",
                "Labelled rows": f"{stats['n_with_historical_outcome']:,}",
                "Observed bads": f"{stats['observed_bad_count']:,}",
            }
        )
    st.dataframe(pd.DataFrame(swaps), hide_index=True, use_container_width=True)
    st.caption("Average model PD and the expected bad count are model-expected. The observed bad rate is historical.")
    causes = pd.DataFrame(credit["swap_causes"])
    st.dataframe(causes, hide_index=True, use_container_width=True)
    band = credit["bureau_580_to_599"]
    st.markdown(
        f"Bureau 580–599: {band['rows']:,} applications, "
        f"{band['decision_changes']:,} decision changes, "
        f"mean model PD {band['mean_model_pd']:.4f}, minimum {band['min_model_pd']:.4f}."
    )
    fraud = payload["fraud"]["versions"]
    fraud_rows = []
    for version in ("fraud_v1", "fraud_v2"):
        row = fraud[version]
        fraud_rows.append(
            {
                "Version": version,
                "Alert rate": f"{row['alert_rate']:.4f}",
                "Fraud caught": f"{row['fraud_caught_rate']:.4f}",
                "Fraud value caught (INR)": f"{row['fraud_value_caught']:,.0f}",
                "False positives": f"{row['false_positives']:,}",
                "Genuine blocked": f"{row['genuine_transactions_blocked']:,}",
                "Genuine challenged": f"{row['genuine_transactions_challenged']:,}",
                "Grocery bursts blocked": f"{row['grocery_burst_blocked']:,}",
            }
        )
    st.dataframe(pd.DataFrame(fraud_rows), hide_index=True, use_container_width=True)
    status = "passed" if golden["passed"] else "failed"
    st.markdown(f"Golden tests: **{status}**. {golden['cases']} cases. {golden['summary']}.")
    report = (DATA / "strategy_change_v1_to_v2.md").read_bytes()
    st.download_button(
        "Download the full validation report",
        data=report,
        file_name="strategy_change_v1_to_v2.md",
        mime="text/markdown",
    )
    st.caption("The same report is in the repo at reports/strategy_change_v1_to_v2.md.")


def _try_it() -> None:
    st.markdown(
        "This form calls the same engine and the same saved model as the batch. "
        "Debt-to-income is computed with the project formula: existing debt × 3%, "
        "plus a 16% annuity on a personal loan or 5% of the limit on a card, "
        "divided by monthly income."
    )
    loan, card = st.tabs(["Loan application", "Card transaction"])
    with loan:
        _loan_form()
    with card:
        _card_form()


def _loan_form() -> None:
    with st.form("loan"):
        left, right = st.columns(2)
        with left:
            age = st.number_input("Age", min_value=18, max_value=80, value=35)
            income = st.number_input("Annual income (INR)", min_value=0, value=600_000, step=10_000)
            existing_debt = st.number_input("Existing debt (INR)", min_value=0, value=380_000, step=10_000)
            bureau = st.number_input("Bureau score", min_value=300, max_value=900, value=720)
            product = st.selectbox("Product", ["personal loan", "credit card"])
        with right:
            amount = st.number_input("Amount asked (INR)", min_value=1000, value=100_000, step=1000)
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=60, value=36)
            delinquencies = st.number_input("Delinquencies in 24 months", min_value=0, max_value=20, value=0)
            enquiries = st.number_input("Enquiries in 6 months", min_value=0, max_value=20, value=1)
            utilisation = st.number_input("Utilisation", min_value=0.0, max_value=1.5, value=0.40, step=0.01)
            limit = st.number_input(
                "Existing credit limit (INR)",
                min_value=0,
                value=200_000,
                step=10_000,
                help="Zero marks a thin file, which uses the tighter score band.",
            )
        strategy_name = st.radio("Strategy", ["credit_v1", "credit_v2"], horizontal=True)
        submitted = st.form_submit_button("Score application")
    if not submitted:
        return
    record = {
        "application_id": "DEMO",
        "age": int(age),
        "income": float(income),
        "existing_debt": float(existing_debt),
        "bureau_score": float(bureau),
        "product": product,
        "requested_amount": float(amount),
        "tenure_months": int(tenure),
        "num_delinquencies_24m": int(delinquencies),
        "num_enquiries_6m": int(enquiries),
        "utilisation": float(utilisation),
        "total_credit_limit": float(limit),
    }
    from src.data_gen.performance import _debt_to_income

    record["debt_to_income"] = float(_debt_to_income(pd.DataFrame([record]))[0])
    result = _score(record, strategy_name)
    _show_decision(result, extra=f"Debt-to-income used by the strategy: {record['debt_to_income']:.4f}")


def _card_form() -> None:
    with st.form("card"):
        left, right = st.columns(2)
        with left:
            amount = st.number_input("Amount (INR)", min_value=0.0, value=850.0, step=50.0)
            count_1h = st.number_input("Transactions in the past hour", min_value=0, value=0)
            amount_1h = st.number_input("Amount in the past hour (INR)", min_value=0.0, value=0.0, step=50.0)
            count_24h = st.number_input("Transactions in 24 hours", min_value=0, value=1)
            amount_24h = st.number_input("Amount in 24 hours (INR)", min_value=0.0, value=850.0, step=50.0)
        with right:
            average = st.number_input("Average amount, 30 days (INR)", min_value=0.0, value=900.0, step=50.0)
            new_device = st.checkbox("New device")
            foreign = st.checkbox("Foreign country")
            night = st.checkbox("Night (before 05:00)")
            strategy_name = st.radio("Strategy", ["fraud_v1", "fraud_v2"], horizontal=True)
        submitted = st.form_submit_button("Score transaction")
    if not submitted:
        return
    ratio = None if average <= 0 else float(amount) / float(average)
    record = {
        "txn_id": "DEMO",
        "amount": float(amount),
        "txn_ts": "2026-09-15 02:10:00" if night else "2026-09-15 14:10:00",
        "txn_count_1h": int(count_1h),
        "txn_amount_1h": float(amount_1h),
        "txn_count_24h": int(count_24h),
        "txn_amount_24h": float(amount_24h),
        "avg_amount_30d": float(average) if average > 0 else None,
        "amount_vs_avg_30d": ratio,
        "is_new_device": int(new_device),
        "is_foreign_country": int(foreign),
        "is_night": int(night),
    }
    result = _score(record, strategy_name)
    ratio_text = "not used (no 30-day average)" if ratio is None else f"{ratio:.2f}"
    _show_decision(result, extra=f"Amount vs 30-day average: {ratio_text}")


def _score(record: dict, strategy_name: str) -> dict:
    from src.decision_engine.engine import decide_one

    models, strategies = load_runtime()
    strategy = strategies[strategy_name]
    return decide_one(record, strategy, models[strategy.model_version])


def _show_decision(result: dict, extra: str) -> None:
    columns = st.columns(3)
    columns[0].metric("Decision", result["decision"])
    columns[1].metric("Model PD", "n/a" if result["score"] is None else f"{result['score']:.4f}")
    limit = result["credit_limit"]
    columns[2].metric("Limit", "n/a" if limit is None else _rupees(limit))
    st.write("Reason codes:", ", ".join(result["reason_codes"]) or "none")
    st.caption(f"{extra} · strategy {result['strategy_version']} · model {result['model_version']}")


def _style(figure: go.Figure, height: int = 380) -> None:
    figure.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=56, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )


def _rupees(amount: float | None) -> str:
    if amount is None:
        return "n/a"
    return f"₹{amount:,.0f}"


if __name__ == "__main__":
    main()
else:
    main()
