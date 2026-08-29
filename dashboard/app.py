"""Streamlit dashboard for the Credit Risk FastAPI service."""

from __future__ import annotations

import os
import json
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.api_client import (
    API_BASE_URL,
    CreditRiskAPIClient,
    CreditRiskAPIError,
    CreditRiskAPIUnavailable,
)
from dashboard.utils import (
    DECISION_ORDER,
    DEFAULT_GRADE_ORDER,
    batch_count,
    chunk_records,
    format_pd,
    format_score,
    merge_predictions,
    monitoring_summary,
    normalize_records_for_json,
    ordered_counts,
    batch_limit_from_openapi,
    portfolio_kpis,
    required_applicant_fields,
    distribution_comparison_frame,
    split_optional_labels,
    top_feature_psi,
    validate_portfolio_columns,
)


FIELD_UI: dict[str, dict[str, Any]] = {
    "person_age": {"label": "Applicant Age", "default": 35.0, "step": 1.0},
    "person_income": {"label": "Annual Income", "default": 65000.0, "step": 1000.0},
    "person_home_ownership": {
        "label": "Home Ownership",
        "options": ["RENT", "MORTGAGE", "OWN", "OTHER"],
    },
    "person_emp_length": {
        "label": "Employment Length",
        "default": 5.0,
        "step": 1.0,
        "nullable": True,
        "help": "Choose Unknown / Missing to send null to the API.",
    },
    "loan_intent": {
        "label": "Loan Intent",
        "options": [
            "PERSONAL",
            "EDUCATION",
            "MEDICAL",
            "VENTURE",
            "DEBTCONSOLIDATION",
            "HOMEIMPROVEMENT",
        ],
    },
    "loan_amnt": {"label": "Loan Amount", "default": 18000.0, "step": 500.0},
    "cb_person_default_on_file": {"label": "Previous Default on File", "options": ["N", "Y"]},
    "cb_person_cred_hist_length": {"label": "Credit History Length", "default": 4.0, "step": 1.0},
    "marital_status": {"label": "Marital Status", "options": ["single", "married", "divorced", "widowed"]},
    "education_level": {"label": "Education Level", "options": ["High School", "Bachelor", "Master", "PhD"]},
    "employment_type": {
        "label": "Employment Type",
        "options": ["full_time", "part_time", "self_employed", "unemployed"],
    },
    "loan_term_months": {"label": "Loan Term Months", "default": 36.0, "step": 12.0},
    "loan_to_income_ratio": {
        "label": "Loan-to-Income Ratio",
        "default": 0.276923,
        "step": 0.01,
        "help": "Loan amount divided by annual income.",
    },
    "other_debt": {"label": "Other Debt", "default": 20000.0, "step": 500.0},
    "debt_to_income_ratio": {
        "label": "Debt-to-Income Ratio",
        "default": 0.50,
        "step": 0.01,
    },
    "open_accounts": {"label": "Open Accounts", "default": 6.0, "step": 1.0},
    "credit_utilization_ratio": {
        "label": "Credit Utilization Ratio",
        "default": 0.85,
        "step": 0.01,
        "help": "Share of available credit currently used.",
    },
    "past_delinquencies": {"label": "Past Delinquencies", "default": 1.0, "step": 1.0},
}


def get_client() -> CreditRiskAPIClient:
    return CreditRiskAPIClient(base_url=os.getenv("CREDIT_RISK_API_URL", API_BASE_URL))


@st.cache_data(ttl=30)
def fetch_health(base_url: str) -> dict[str, Any] | None:
    try:
        return CreditRiskAPIClient(base_url=base_url).health()
    except CreditRiskAPIError:
        return None


@st.cache_data(ttl=60)
def fetch_model_info(base_url: str) -> dict[str, Any] | None:
    try:
        return CreditRiskAPIClient(base_url=base_url).model_info()
    except CreditRiskAPIError:
        return None


@st.cache_data(ttl=60)
def fetch_openapi(base_url: str) -> dict[str, Any] | None:
    try:
        return CreditRiskAPIClient(base_url=base_url).openapi()
    except CreditRiskAPIError:
        return None


def main() -> None:
    st.set_page_config(
        page_title="Credit Risk Dashboard",
        page_icon="CR",
        layout="wide",
    )

    base_url = os.getenv("CREDIT_RISK_API_URL", API_BASE_URL)
    client = CreditRiskAPIClient(base_url=base_url)
    health = fetch_health(base_url)
    model_info = fetch_model_info(base_url) if health else None
    openapi_schema = fetch_openapi(base_url) if health else None

    render_sidebar(base_url, health)

    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Applicant Scoring", "Portfolio Analysis", "Monitoring", "Model Information", "About / Limitations"],
    )

    if page == "Home":
        render_home(health, model_info)
    elif page == "Applicant Scoring":
        render_applicant_scoring(client, health, openapi_schema)
    elif page == "Portfolio Analysis":
        render_portfolio(client, health, openapi_schema, model_info)
    elif page == "Monitoring":
        render_monitoring(client, health, openapi_schema)
    elif page == "Model Information":
        render_model_information(model_info)
    else:
        render_limitations()


def render_sidebar(base_url: str, health: dict[str, Any] | None) -> None:
    st.sidebar.title("Credit Risk")
    st.sidebar.caption(f"API: `{base_url}`")
    if health:
        st.sidebar.success("API Status: Healthy")
        st.sidebar.write(f"Model Loaded: {'Yes' if health.get('model_loaded') else 'No'}")
        st.sidebar.write(f"Policy: {health.get('policy_status', '-')}")
        st.sidebar.write(f"Calibration: {health.get('calibration_method', '-')}")
    else:
        st.sidebar.error("Credit Risk API is currently unavailable.")
        st.sidebar.code("python -m uvicorn api.main:app --reload")


def render_home(health: dict[str, Any] | None, model_info: dict[str, Any] | None) -> None:
    st.title("Credit Risk Scoring & Loan Decision System")
    st.write(
        "Applicant Data -> Credit Risk Model -> PD -> Internal Credit Score -> Risk Grade -> Decision"
    )

    if not health:
        st.warning("Credit Risk API is currently unavailable. Start it with:")
        st.code("python -m uvicorn api.main:app --reload")
        return

    cols = st.columns(4)
    cols[0].metric("Champion Model", model_info.get("model_name", "-") if model_info else "-")
    cols[1].metric("Calibration", health.get("calibration_method", "-"))
    cols[2].metric("Model Version", health.get("model_version", "-"))
    cols[3].metric("Policy Status", health.get("policy_status", "-"))

    if model_info and model_info.get("locked_test_metrics"):
        st.subheader("Locked Test Metrics")
        metrics = model_info["locked_test_metrics"]
        metric_cols = st.columns(4)
        metric_cols[0].metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
        metric_cols[1].metric("PR-AUC", f"{metrics['pr_auc']:.4f}")
        metric_cols[2].metric("KS", f"{metrics['ks']:.4f}")
        metric_cols[3].metric("Brier", f"{metrics['brier']:.4f}")

    st.subheader("System Flow")
    flow_cols = st.columns(7)
    for col, label in zip(
        flow_cols,
        ["Applicant", "FastAPI", "Frozen Model", "PD", "Score", "Grade", "Decision"],
    ):
        col.info(label)


def render_applicant_scoring(
    client: CreditRiskAPIClient,
    health: dict[str, Any] | None,
    openapi_schema: dict[str, Any] | None,
) -> None:
    st.title("Applicant Scoring")
    if not health:
        st.warning("Credit Risk API is currently unavailable. Prediction is disabled.")
        st.code("python -m uvicorn api.main:app --reload")
        return

    fields = required_applicant_fields(openapi_schema or {})
    if not fields:
        st.error("Unable to fetch ApplicantRequest schema from the API.")
        return

    with st.form("applicant_form"):
        applicant = render_applicant_form(fields)
        submitted = st.form_submit_button("Assess Credit Risk", type="primary")

    if submitted:
        with st.spinner("Calling Credit Risk API..."):
            try:
                result = client.predict(applicant)
            except CreditRiskAPIError as exc:
                st.error(str(exc))
                return
        st.session_state["latest_applicant_result"] = result

    result = st.session_state.get("latest_applicant_result")
    if result:
        render_prediction_result(result)


def render_applicant_form(fields: list[str]) -> dict[str, Any]:
    applicant: dict[str, Any] = {}
    left, right = st.columns(2)
    for idx, field in enumerate(fields):
        config = FIELD_UI.get(field, {"label": field.replace("_", " ").title(), "default": 0.0})
        container = left if idx % 2 == 0 else right
        label = config["label"]
        help_text = config.get("help")

        if "options" in config:
            applicant[field] = container.selectbox(
                label,
                options=config["options"],
                key=f"field_{field}",
                help=help_text,
            )
        elif config.get("nullable"):
            missing = container.checkbox(
                f"{label}: Unknown / Missing",
                value=False,
                key=f"missing_{field}",
                help=help_text,
            )
            applicant[field] = None if missing else container.number_input(
                label,
                value=float(config.get("default", 0.0)),
                step=float(config.get("step", 1.0)),
                key=f"field_{field}",
            )
        else:
            applicant[field] = container.number_input(
                label,
                value=float(config.get("default", 0.0)),
                step=float(config.get("step", 1.0)),
                key=f"field_{field}",
                help=help_text,
            )
    return applicant


def render_prediction_result(result: dict[str, Any]) -> None:
    st.subheader("Assessment Result")
    cols = st.columns(4)
    cols[0].metric("Probability of Default", format_pd(result["pd"]))
    cols[1].metric("Internal Credit Score", format_score(result["credit_score"]))
    cols[2].metric("Risk Grade", result["risk_grade"])
    cols[3].metric("Decision", result["decision"])

    st.progress(min(max(float(result["pd"]), 0.0), 1.0), text="Calibrated Probability of Default")

    decision = result["decision"]
    if decision == "APPROVE":
        st.success("Decision: APPROVE")
    elif decision == "MANUAL_REVIEW":
        st.warning("Decision: MANUAL_REVIEW")
    else:
        st.error("Decision: REJECT")

    if result.get("explanation_status") == "explanation_unavailable":
        st.info("Risk-driver explanation is not currently available in the production inference layer.")


def render_portfolio(
    client: CreditRiskAPIClient,
    health: dict[str, Any] | None,
    openapi_schema: dict[str, Any] | None,
    model_info: dict[str, Any] | None,
) -> None:
    st.title("Portfolio Analysis")
    if not health:
        st.warning("Credit Risk API is currently unavailable. Portfolio scoring is disabled.")
        st.code("python -m uvicorn api.main:app --reload")
        return

    fields = required_applicant_fields(openapi_schema or {})
    if not fields:
        st.error("Unable to fetch ApplicantRequest schema from the API.")
        return

    uploaded = st.file_uploader("Upload applicant CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV containing the API ApplicantRequest fields.")
        return

    df = pd.read_csv(uploaded)
    st.write(f"Rows: {len(df):,} | Columns: {df.shape[1]:,}")
    st.dataframe(df.head(20), use_container_width=True)

    missing = validate_portfolio_columns(df, fields)
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        return

    if st.button("Score Portfolio", type="primary"):
        with st.spinner("Scoring portfolio through FastAPI..."):
            try:
                predictions = score_portfolio(
                    client,
                    df.loc[:, fields],
                    chunk_size=batch_limit_from_openapi(openapi_schema or {}),
                )
            except CreditRiskAPIError as exc:
                st.error(str(exc))
                return
        st.session_state["latest_portfolio"] = merge_predictions(df, predictions)

    scored = st.session_state.get("latest_portfolio")
    if scored is not None:
        render_portfolio_results(scored, model_info)


def score_portfolio(
    client: CreditRiskAPIClient,
    df: pd.DataFrame,
    chunk_size: int,
) -> list[dict[str, Any]]:
    records = normalize_records_for_json(df.to_dict(orient="records"))
    chunks = chunk_records(records, chunk_size)
    predictions: list[dict[str, Any]] = []
    for chunk in chunks:
        response = client.predict_batch(chunk)
        predictions.extend(response["predictions"])
    return predictions


def render_portfolio_results(scored: pd.DataFrame, model_info: dict[str, Any] | None) -> None:
    st.subheader("Portfolio KPIs")
    kpis = portfolio_kpis(scored)
    cols = st.columns(4)
    cols[0].metric("Total Applicants", f"{kpis['total_applicants']:,}")
    cols[1].metric("Mean PD", format_pd(kpis["mean_pd"]))
    cols[2].metric("Median PD", format_pd(kpis["median_pd"]))
    cols[3].metric("Avg Credit Score", format_score(kpis["average_credit_score"]))

    cols = st.columns(3)
    cols[0].metric("Approval Rate", format_pd(kpis["approval_rate"]))
    cols[1].metric("Manual Review Rate", format_pd(kpis["manual_review_rate"]))
    cols[2].metric("Rejection Rate", format_pd(kpis["rejection_rate"]))

    st.subheader("Distributions")
    chart_cols = st.columns(2)
    chart_cols[0].plotly_chart(px.histogram(scored, x="pd", nbins=30, title="PD Distribution"), use_container_width=True)
    chart_cols[1].plotly_chart(px.histogram(scored, x="credit_score", nbins=30, title="Credit Score Distribution"), use_container_width=True)

    grade_order = model_info.get("risk_grades", DEFAULT_GRADE_ORDER) if model_info else DEFAULT_GRADE_ORDER
    grade_counts = ordered_counts(scored["risk_grade"], grade_order).reset_index()
    grade_counts.columns = ["risk_grade", "count"]
    decision_counts = ordered_counts(scored["decision"], DECISION_ORDER).reset_index()
    decision_counts.columns = ["decision", "count"]

    chart_cols = st.columns(2)
    chart_cols[0].plotly_chart(px.bar(grade_counts, x="risk_grade", y="count", title="Risk Grade Distribution"), use_container_width=True)
    chart_cols[1].plotly_chart(px.bar(decision_counts, x="decision", y="count", title="Decision Distribution"), use_container_width=True)

    st.subheader("Scored Portfolio")
    st.dataframe(scored, use_container_width=True)
    st.download_button(
        "Download scored CSV",
        data=scored.to_csv(index=False).encode("utf-8"),
        file_name="credit_risk_scored_portfolio.csv",
        mime="text/csv",
    )


def render_monitoring(
    client: CreditRiskAPIClient,
    health: dict[str, Any] | None,
    openapi_schema: dict[str, Any] | None,
) -> None:
    st.title("Monitoring")
    if not health:
        st.warning("Credit Risk API is currently unavailable. Monitoring is disabled.")
        st.code("python -m uvicorn api.main:app --reload")
        return

    fields = required_applicant_fields(openapi_schema or {})
    if not fields:
        st.error("Unable to fetch ApplicantRequest schema from the API.")
        return

    try:
        reference = client.monitoring_reference()
        st.caption(
            "Reference: Development population "
            f"| Rows: {reference.get('reference_rows', '-')} "
            f"| Model: {reference.get('model_version', '-')}"
        )
    except CreditRiskAPIError as exc:
        st.warning(str(exc))

    uploaded = st.file_uploader("Upload current portfolio CSV", type=["csv"], key="monitoring_upload")
    if uploaded is None:
        st.info("Upload a CSV containing ApplicantRequest fields. Include loan_status only when realized outcomes exist.")
        return

    df = pd.read_csv(uploaded)
    st.write(f"Rows: {len(df):,} | Columns: {df.shape[1]:,}")
    st.dataframe(df.head(20), use_container_width=True)

    missing = validate_portfolio_columns(df, fields)
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        return

    if st.button("Analyze Monitoring Batch", type="primary"):
        try:
            applicant_frame, labels = split_optional_labels(df, fields)
        except ValueError as exc:
            st.error(str(exc))
            return
        records = normalize_records_for_json(applicant_frame.to_dict(orient="records"))
        with st.spinner("Analyzing drift through FastAPI..."):
            try:
                report = client.monitoring_analyze(records, labels=labels)
            except CreditRiskAPIError as exc:
                st.error(str(exc))
                return
        st.session_state["latest_monitoring_report"] = report

    report = st.session_state.get("latest_monitoring_report")
    if report:
        render_monitoring_report(report)


def render_monitoring_report(report: dict[str, Any]) -> None:
    summary = monitoring_summary(report)
    st.subheader("Monitoring Summary")
    cols = st.columns(4)
    cols[0].metric("Overall Status", summary["monitoring_status"])
    cols[1].metric("Current Rows", f"{summary['current_rows']:,}")
    cols[2].metric("Alert Features", summary["alert_feature_count"])
    cols[3].metric("Warning Features", summary["warning_feature_count"])

    cols = st.columns(4)
    cols[0].metric("PD PSI", f"{summary['pd_psi']:.4f}" if summary["pd_psi"] is not None else "-")
    cols[1].metric("Score PSI", f"{summary['score_psi']:.4f}" if summary["score_psi"] is not None else "-")
    cols[2].metric("Grade PSI", f"{summary['risk_grade_psi']:.4f}" if summary["risk_grade_psi"] is not None else "-")
    cols[3].metric("Decision PSI", f"{summary['decision_psi']:.4f}" if summary["decision_psi"] is not None else "-")

    st.subheader("Feature Drift")
    feature_rows = report.get("feature_drift", [])
    feature_table = pd.DataFrame(feature_rows)
    display_cols = [
        "feature",
        "feature_type",
        "psi",
        "reference_missing_rate",
        "current_missing_rate",
        "missing_rate_delta",
        "status",
    ]
    st.dataframe(feature_table.loc[:, [col for col in display_cols if col in feature_table.columns]], use_container_width=True)

    chart_cols = st.columns(2)
    top_psi = top_feature_psi(feature_rows, n=10)
    if not top_psi.empty:
        chart_cols[0].plotly_chart(px.bar(top_psi, x="psi", y="feature", color="status", orientation="h", title="Top Feature PSI"), use_container_width=True)

    pd_frame = distribution_comparison_frame(report.get("pd_drift", {}), "pd_bucket")
    if not pd_frame.empty:
        pd_long = pd_frame.melt(id_vars="pd_bucket", value_vars=["reference_pct", "current_pct"], var_name="population", value_name="pct")
        chart_cols[1].plotly_chart(px.bar(pd_long, x="pd_bucket", y="pct", color="population", barmode="group", title="Reference vs Current PD Distribution"), use_container_width=True)

    chart_cols = st.columns(2)
    grade_frame = distribution_comparison_frame(report.get("risk_grade_drift", {}), "risk_grade")
    if not grade_frame.empty:
        grade_long = grade_frame.melt(id_vars="risk_grade", value_vars=["reference_pct", "current_pct"], var_name="population", value_name="pct")
        chart_cols[0].plotly_chart(px.bar(grade_long, x="risk_grade", y="pct", color="population", barmode="group", title="Reference vs Current Risk Grade"), use_container_width=True)

    decision_frame = distribution_comparison_frame(report.get("decision_drift", {}), "decision")
    if not decision_frame.empty:
        decision_long = decision_frame.melt(id_vars="decision", value_vars=["reference_pct", "current_pct"], var_name="population", value_name="pct")
        chart_cols[1].plotly_chart(px.bar(decision_long, x="decision", y="pct", color="population", barmode="group", title="Reference vs Current Decision"), use_container_width=True)

    render_monitoring_performance(report.get("performance", {}))
    st.download_button(
        "Download monitoring_report.json",
        data=json.dumps(report, indent=2).encode("utf-8"),
        file_name="monitoring_report.json",
        mime="application/json",
    )


def render_monitoring_performance(performance: dict[str, Any]) -> None:
    st.subheader("Performance Monitoring")
    status_value = performance.get("performance_status")
    if status_value != "available":
        st.info("Performance monitoring requires realized loan outcomes.")
        if status_value and status_value != "labels_unavailable":
            st.write(status_value)
        return

    cols = st.columns(4)
    cols[0].metric("ROC-AUC", f"{performance['roc_auc']:.4f}")
    cols[1].metric("PR-AUC", f"{performance['pr_auc']:.4f}")
    cols[2].metric("KS", f"{performance['ks']:.4f}")
    cols[3].metric("Gini", f"{performance['gini']:.4f}")

    cols = st.columns(4)
    cols[0].metric("Brier", f"{performance['brier']:.4f}")
    cols[1].metric("Observed Default Rate", format_pd(performance["observed_default_rate"]))
    cols[2].metric("Mean Predicted PD", format_pd(performance["mean_predicted_pd"]))
    cols[3].metric("Calibration Gap", format_pd(performance["calibration_gap"]))


def render_model_information(model_info: dict[str, Any] | None) -> None:
    st.title("Model Information")
    if not model_info:
        st.warning("Unable to fetch model metadata from the API.")
        st.code("python -m uvicorn api.main:app --reload")
        return

    cols = st.columns(4)
    cols[0].metric("Model Family", model_info.get("model_family", "-"))
    cols[1].metric("Model Version", model_info.get("model_version", "-"))
    cols[2].metric("Calibration", model_info.get("calibration_method", "-"))
    cols[3].metric("Policy Status", model_info.get("policy_status", "-"))

    st.write(f"Raw feature count: **{model_info.get('raw_feature_count', '-')}**")
    score_range = model_info.get("score_range", {})
    st.write(f"Internal Credit Score range: **{score_range.get('min', '-')} - {score_range.get('max', '-')}**")
    st.write("Risk grades: " + ", ".join(model_info.get("risk_grades", [])))

    if model_info.get("locked_test_metrics"):
        st.subheader("Locked Test Metrics")
        st.dataframe(pd.DataFrame([model_info["locked_test_metrics"]]), use_container_width=True)

    render_limitations()


def render_limitations() -> None:
    st.title("About / Limitations")
    st.markdown(
        """
        - Portfolio/demo project, not a production-approved bank model.
        - Predictions are decision-support outputs, not automatic real-world lending authority.
        - Thresholds are based on a simulated business policy.
        - Dataset has no true production timeline.
        - Internal Credit Score is a project score, not FICO or an official bureau score.
        - Risk drivers are currently unavailable in the production inference layer.
        - Monitoring is a simulation against the Development reference population, not real-time production monitoring.
        - PSI thresholds are configurable heuristics; drift signals require human investigation before retraining decisions.
        """
    )


if __name__ == "__main__":
    main()
