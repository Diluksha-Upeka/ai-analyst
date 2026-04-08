import importlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from analytics.data_health import compute_data_health as calculate_data_health
from analytics.root_cause import (
    compute_root_cause_period as calculate_root_cause_period,
    detect_date_columns as find_date_columns,
    get_period_labels as build_period_labels,
)
from analytics.what_if import compute_what_if_simulation as calculate_what_if_simulation


logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ai_data_analyst")

BLOCKED_QUERY_PATTERNS = [
    "__import__",
    "import os",
    "import subprocess",
    "os.system",
    "subprocess",
    "exec(",
    "eval(",
    "open(",
    "rm -rf",
    "del /f",
    "cmd.exe",
    "powershell",
]


@st.cache_data
def load_csv(file):
    return pd.read_csv(file)


@st.cache_data
def compute_data_health_cached(_df):
    return calculate_data_health(_df)


@st.cache_data
def detect_date_columns_cached(_df):
    return find_date_columns(_df)


@st.cache_data
def get_period_labels_cached(_df, date_column, period_grain):
    return build_period_labels(_df, date_column, period_grain)


@st.cache_data
def compute_root_cause_period_cached(
    _df,
    metric_column,
    dimension_column,
    date_column,
    period_grain,
    baseline_period,
    comparison_period,
):
    return calculate_root_cause_period(
        _df,
        metric_column,
        dimension_column,
        date_column,
        period_grain,
        baseline_period,
        comparison_period,
    )


@st.cache_data
def compute_what_if_simulation_cached(_df, target_metric, driver_adjustments, agg_method):
    return calculate_what_if_simulation(_df, target_metric, driver_adjustments, agg_method)


def is_query_safe(user_question):
    lowered = user_question.lower()
    for pattern in BLOCKED_QUERY_PATTERNS:
        if pattern in lowered:
            return False
    return True


def initialize_file_session(file_name):
    if "analysis_journal" not in st.session_state:
        st.session_state.analysis_journal = []

    if "current_file" not in st.session_state or st.session_state.current_file != file_name:
        st.session_state.current_file = file_name
        st.session_state.messages = []
        st.session_state.analysis_journal = []
        st.session_state.pop("session_id", None)
        st.session_state.pop("last_scenario_payload", None)


def add_journal_entry(entry_type, title, summary, context):
    if "analysis_journal" not in st.session_state:
        st.session_state.analysis_journal = []

    st.session_state.analysis_journal.append(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "entry_type": entry_type,
            "title": title,
            "summary": summary,
            "context": context,
        }
    )


def get_plot_path():
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex

    return artifacts_dir / f"plot_{st.session_state.session_id}.png"


@st.cache_resource
def create_agent(_df, api_key, file_name, model_name):
    try:
        from langchain_groq import ChatGroq
        from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
        try:
            from langchain_classic.agents.agent_types import AgentType
        except ModuleNotFoundError:  # Back-compat with older LangChain layouts
            AgentType = importlib.import_module("langchain.agents.agent_types").AgentType
    except Exception as exc:
        raise RuntimeError(
            "LangChain dependencies are incompatible in this Python environment. "
            "Run the app from the project virtual environment."
        ) from exc

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0,
    )
    return create_pandas_dataframe_agent(
        llm,
        _df,
        verbose=False,
        allow_dangerous_code=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=15,
    )


def parse_agent_response(agent, enhanced_query):
    if hasattr(agent, "invoke"):
        response_payload = agent.invoke({"input": enhanced_query})
        if isinstance(response_payload, dict):
            return response_payload.get("output") or str(response_payload)
        return str(response_payload)

    return agent.run(enhanced_query)


def render_data_health_section(df):
    st.write("### Data Health Check")
    health = compute_data_health_cached(df)

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric("Health Score", f"{health['health_score']}/100")
        st.caption(f"Status: {health['health_status']}")

    with metric_col2:
        st.metric("Missing Cells", f"{health['missing_cells']:,}")

    with metric_col3:
        st.metric("Duplicate Rows", f"{health['duplicate_rows']:,}")

    with metric_col4:
        st.metric("Outlier Values", f"{health['outlier_cells']:,}")

    with st.expander("Missing Values Details", expanded=False):
        if health["missing_cells"] == 0:
            st.success("No missing values detected.")
        else:
            st.warning(
                f"Found {health['missing_cells']:,} missing values across {health['missing_columns']} columns."
            )
            st.dataframe(health["missing_df"], width="stretch")

    with st.expander("Duplicate Rows Details", expanded=False):
        if health["duplicate_rows"] == 0:
            st.success("No duplicate rows detected.")
        else:
            st.warning(f"Found {health['duplicate_rows']:,} duplicate rows.")

    with st.expander("Type Issues Details", expanded=False):
        if health["type_issue_count"] == 0:
            st.success("No likely type issues detected.")
        else:
            st.warning(f"Found {health['type_issue_count']} likely type issues.")
            st.dataframe(health["type_issue_df"], width="stretch")

    with st.expander("Outlier Details", expanded=False):
        if health["outlier_cells"] == 0:
            st.success("No outliers detected with the IQR rule.")
        else:
            st.warning(f"Detected {health['outlier_cells']:,} outlier values.")
            st.dataframe(health["outlier_df"], width="stretch")

    with st.expander("Health Score Breakdown", expanded=False):
        st.dataframe(health["penalty_df"], width="stretch")

    if st.button("Save Data Health Snapshot", key="save_data_health_snapshot"):
        add_journal_entry(
            entry_type="data_health",
            title="Data Health Snapshot",
            summary=f"Score {health['health_score']}/100 with {health['missing_cells']} missing cells and {health['duplicate_rows']} duplicates.",
            context={
                "health_score": health["health_score"],
                "health_status": health["health_status"],
                "missing_cells": health["missing_cells"],
                "duplicate_rows": health["duplicate_rows"],
                "outlier_cells": health["outlier_cells"],
            },
        )
        st.success("Saved Data Health snapshot to Analysis Journal.")


def render_quick_insights_section(df):
    st.write("### Quick Insights")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Dataset Summary"):
            st.write(df.describe())

    with col2:
        if st.button("Correlation Matrix"):
            st.write(df.corr(numeric_only=True))


def render_root_cause_section(df):
    st.write("### Root Cause Explorer")
    st.caption("Explain what drove metric change between two time periods.")

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    date_columns = detect_date_columns_cached(df)

    if not numeric_columns:
        st.info("Root Cause Explorer needs at least one numeric metric column.")
        return

    if not date_columns:
        st.info("No date-like column detected. Add a date column to run period-over-period root cause analysis.")
        return

    root_metric = st.selectbox("Metric to explain", numeric_columns, key="root_metric")

    dimension_candidates = [
        column
        for column in df.columns
        if column != root_metric
        and column not in date_columns
        and (
            pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
            or pd.api.types.is_bool_dtype(df[column])
            or (
                pd.api.types.is_integer_dtype(df[column])
                and 2 <= df[column].nunique(dropna=True) <= 40
            )
        )
    ]

    if not dimension_candidates:
        st.info("Root Cause Explorer needs at least one categorical column for breakdown.")
        return

    root_dimension = st.selectbox("Break down change by", dimension_candidates, key="root_dimension")
    root_date = st.selectbox("Date column", date_columns, key="root_date")
    period_grain = st.selectbox(
        "Period grain",
        ["Monthly", "Quarterly", "Weekly", "Daily"],
        key="root_period_grain",
    )

    period_labels = get_period_labels_cached(df, root_date, period_grain)
    if len(period_labels) < 2:
        st.warning("Need at least two distinct periods to compare.")
        return

    default_comparison_index = len(period_labels) - 1
    default_baseline_index = max(0, default_comparison_index - 1)

    baseline_period = st.selectbox(
        "Baseline period",
        period_labels,
        index=default_baseline_index,
        key="root_baseline_period",
    )

    comparison_candidates = [period for period in period_labels if period != baseline_period]
    if not comparison_candidates:
        st.warning("Select a different baseline period.")
        return

    default_comparison_period = period_labels[default_comparison_index]
    if default_comparison_period in comparison_candidates:
        comparison_default_index = comparison_candidates.index(default_comparison_period)
    else:
        comparison_default_index = len(comparison_candidates) - 1

    comparison_period = st.selectbox(
        "Comparison period",
        comparison_candidates,
        index=comparison_default_index,
        key="root_comparison_period",
    )

    top_n = st.slider(
        "Top drivers to display",
        min_value=3,
        max_value=15,
        value=8,
        key="root_top_n",
    )

    root_result = compute_root_cause_period_cached(
        df,
        root_metric,
        root_dimension,
        root_date,
        period_grain,
        baseline_period,
        comparison_period,
    )

    if root_result is None or root_result["contribution_df"].empty:
        st.warning("Not enough valid rows for this root cause setup.")
        return

    baseline_total = root_result["baseline_total"]
    comparison_total = root_result["comparison_total"]
    net_change = root_result["net_change"]
    pct_change = root_result["pct_change"]

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.metric(f"{baseline_period} total", f"{baseline_total:,.2f}")

    with summary_col2:
        st.metric(f"{comparison_period} total", f"{comparison_total:,.2f}")

    with summary_col3:
        delta_text = "n/a" if pct_change is None else f"{pct_change:+.2f}%"
        st.metric("Net change", f"{net_change:,.2f}", delta=delta_text)

    if net_change > 0:
        st.success(
            f"{root_metric} increased by {net_change:,.2f} from {baseline_period} to {comparison_period}."
        )
    elif net_change < 0:
        st.warning(
            f"{root_metric} decreased by {abs(net_change):,.2f} from {baseline_period} to {comparison_period}."
        )
    else:
        st.info(f"No net change detected for {root_metric} between the selected periods.")

    contribution_df = root_result["contribution_df"]
    top_driver_df = contribution_df.head(top_n)

    st.write("Top Drivers by Absolute Change")
    st.bar_chart(top_driver_df.set_index("dimension")[["change"]], height=320)

    positive_drivers = (
        contribution_df[contribution_df["change"] > 0]
        .sort_values(by="change", ascending=False)
        .head(top_n)
    )
    negative_drivers = (
        contribution_df[contribution_df["change"] < 0]
        .sort_values(by="change", ascending=True)
        .head(top_n)
    )

    driver_col1, driver_col2 = st.columns(2)

    with driver_col1:
        st.write("Positive Drivers")
        if positive_drivers.empty:
            st.caption("No positive drivers in this comparison.")
        else:
            st.dataframe(
                positive_drivers[
                    [
                        "dimension",
                        "baseline_value",
                        "comparison_value",
                        "change",
                        "contribution_pct",
                    ]
                ],
                width="stretch",
            )

    with driver_col2:
        st.write("Negative Drivers")
        if negative_drivers.empty:
            st.caption("No negative drivers in this comparison.")
        else:
            st.dataframe(
                negative_drivers[
                    [
                        "dimension",
                        "baseline_value",
                        "comparison_value",
                        "change",
                        "contribution_pct",
                    ]
                ],
                width="stretch",
            )

    with st.expander("Full Contribution Table", expanded=False):
        st.dataframe(contribution_df, width="stretch")

    save_key = f"save_root_cause_{root_metric}_{root_dimension}_{baseline_period}_{comparison_period}"
    if st.button("Save Root Cause Snapshot", key=save_key):
        top_records = top_driver_df[
            ["dimension", "change", "contribution_pct"]
        ].head(5).to_dict(orient="records")
        add_journal_entry(
            entry_type="root_cause",
            title=f"{root_metric}: {baseline_period} -> {comparison_period}",
            summary=f"Net change {net_change:,.2f} driven by {root_dimension} categories.",
            context={
                "metric": root_metric,
                "dimension": root_dimension,
                "period_grain": period_grain,
                "baseline_period": baseline_period,
                "comparison_period": comparison_period,
                "baseline_total": baseline_total,
                "comparison_total": comparison_total,
                "net_change": net_change,
                "pct_change": pct_change,
                "top_drivers": top_records,
            },
        )
        st.success("Saved Root Cause snapshot to Analysis Journal.")


def render_what_if_section(df):
    st.write("### What-If Simulator")
    st.caption("Test KPI impact by changing one or more numeric drivers.")

    scenario_numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    if len(scenario_numeric_columns) < 2:
        st.info("What-If Simulator needs at least two numeric columns (one metric and one driver).")
        return

    target_metric = st.selectbox(
        "Target metric",
        scenario_numeric_columns,
        key="scenario_target_metric",
    )

    driver_options = [column for column in scenario_numeric_columns if column != target_metric]
    selected_drivers = st.multiselect(
        "Drivers to simulate",
        driver_options,
        default=driver_options[:1],
        max_selections=3,
        key="scenario_drivers",
        help="Choose up to 3 numeric drivers.",
    )

    agg_method = st.selectbox(
        "Aggregation",
        ["Sum", "Mean", "Median"],
        key="scenario_aggregation",
        help="Controls how baseline metric and drivers are summarized before applying scenario changes.",
    )

    if not selected_drivers:
        st.info("Select at least one driver to run a scenario.")
        return

    driver_adjustments = []
    for driver_column in selected_drivers:
        pct_change = st.slider(
            f"{driver_column} change (%)",
            min_value=-50,
            max_value=50,
            value=0,
            step=1,
            key=f"scenario_change_{driver_column}",
        )
        driver_adjustments.append((driver_column, pct_change))

    signature = (
        target_metric,
        tuple(driver_adjustments),
        agg_method,
    )

    if st.button("Run What-If Scenario"):
        scenario_result = compute_what_if_simulation_cached(
            df,
            target_metric,
            tuple(driver_adjustments),
            agg_method,
        )
        st.session_state.last_scenario_payload = {
            "signature": signature,
            "result": scenario_result,
            "target_metric": target_metric,
            "driver_adjustments": list(driver_adjustments),
            "agg_method": agg_method,
        }

    payload = st.session_state.get("last_scenario_payload")
    if not payload or payload.get("signature") != signature:
        st.caption("Run the scenario to view projected impact.")
        return

    scenario_result = payload.get("result")
    if scenario_result is None:
        st.warning("Not enough valid numeric data to compute this scenario.")
        return

    baseline_target = scenario_result["baseline_target"]
    projected_target = scenario_result["projected_target"]
    net_delta = scenario_result["net_delta"]
    projected_pct = scenario_result["projected_pct"]

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.metric("Baseline Target", f"{baseline_target:,.2f}")

    with summary_col2:
        delta_text = "n/a" if projected_pct is None else f"{projected_pct:+.2f}%"
        st.metric("Projected Target", f"{projected_target:,.2f}", delta=delta_text)

    with summary_col3:
        st.metric("Estimated Net Impact", f"{net_delta:,.2f}")

    if net_delta > 0:
        st.success(f"Scenario projects an increase of {net_delta:,.2f} in {target_metric}.")
    elif net_delta < 0:
        st.warning(f"Scenario projects a decrease of {abs(net_delta):,.2f} in {target_metric}.")
    else:
        st.info("Scenario projects no net change in the target metric.")

    impact_df = scenario_result["impact_df"].copy()
    st.write("Driver Impacts")
    st.bar_chart(
        impact_df.set_index("driver")[["predicted_metric_impact"]],
        height=300,
    )
    st.dataframe(impact_df, width="stretch")
    st.caption(
        "Impact is estimated from historical linear relationships (slope) and should be treated as directional guidance."
    )

    if st.button("Save Scenario Snapshot", key="save_scenario_snapshot"):
        add_journal_entry(
            entry_type="what_if",
            title=f"What-If: {target_metric}",
            summary=f"Projected change {net_delta:,.2f} ({'n/a' if projected_pct is None else str(projected_pct) + '%'})",
            context={
                "target_metric": target_metric,
                "aggregation": agg_method,
                "driver_adjustments": list(driver_adjustments),
                "baseline_target": baseline_target,
                "projected_target": projected_target,
                "net_delta": net_delta,
                "projected_pct": projected_pct,
                "driver_impacts": impact_df.to_dict(orient="records"),
            },
        )
        st.success("Saved What-If scenario to Analysis Journal.")


def render_analysis_journal_section():
    st.write("### Analysis Journal")
    st.caption("Day 5: Save and export your analysis trail for reproducibility.")

    journal = st.session_state.get("analysis_journal", [])
    if not journal:
        st.info("No journal entries yet. Save snapshots from Data Health, Root Cause, What-If, or AI chat.")
        return

    journal_df = pd.DataFrame(journal)
    display_df = journal_df[["timestamp_utc", "entry_type", "title", "summary"]].copy()
    display_df = display_df.sort_values(by="timestamp_utc", ascending=False)

    st.dataframe(display_df, width="stretch")

    file_slug = st.session_state.get("current_file", "dataset")
    file_slug = file_slug.replace(".csv", "")
    json_payload = json.dumps(journal, indent=2)
    csv_payload = display_df.to_csv(index=False)

    action_col1, action_col2, action_col3 = st.columns(3)

    with action_col1:
        st.download_button(
            "Download Journal (JSON)",
            data=json_payload,
            file_name=f"analysis_journal_{file_slug}.json",
            mime="application/json",
            key="download_journal_json",
        )

    with action_col2:
        st.download_button(
            "Download Journal (CSV)",
            data=csv_payload,
            file_name=f"analysis_journal_{file_slug}.csv",
            mime="text/csv",
            key="download_journal_csv",
        )

    with action_col3:
        if st.button("Clear Journal", key="clear_analysis_journal"):
            st.session_state.analysis_journal = []
            st.success("Analysis Journal cleared.")


def render_chat_section(df, api_key, model_name, execution_confirmed, uploaded_file_name, plot_path):
    if not api_key:
        st.info("Add your API key in the sidebar to enable AI chat analysis.")
        return

    if not execution_confirmed:
        st.info("Enable advanced AI analysis in the sidebar to run custom questions and chart generation.")

    st.write("### Ask a Question")
    user_question = st.text_input("Example: 'Plot a bar chart of Sales by Region and save it as plot.png'")

    if st.button("Analyze"):
        if not execution_confirmed:
            st.warning("Enable advanced AI analysis in the sidebar before running a question.")
            return

        if not user_question.strip():
            st.warning("Please enter a question first.")
            return

        if not is_query_safe(user_question):
            st.error("This question looks unsafe. Please remove system or file commands and try again.")
            return

        with st.spinner("Analyzing data..."):
            try:
                agent = create_agent(df, api_key, uploaded_file_name, model_name)

                if plot_path.exists():
                    plot_path.unlink()

                safe_plot_path = plot_path.as_posix()
                enhanced_query = (
                    f"{user_question} "
                    f"If you generate a plot, save it as '{safe_plot_path}'. "
                    "Do not use plt.show()."
                )

                logger.info(
                    "Analyze request | file=%s | model=%s | question=%s",
                    uploaded_file_name,
                    model_name,
                    user_question[:200],
                )

                response = parse_agent_response(agent, enhanced_query)
                st.success("Analysis Complete!")
                st.write(response)

                if plot_path.exists():
                    st.image(str(plot_path), caption="Generated Visualization")

                response_preview = response if len(response) <= 240 else response[:240] + "..."
                add_journal_entry(
                    entry_type="ai_chat",
                    title="AI Question",
                    summary=response_preview,
                    context={
                        "model_name": model_name,
                        "question": user_question,
                        "plot_generated": bool(plot_path.exists()),
                    },
                )

            except Exception as e:
                logger.exception(
                    "Analyze failed | file=%s | model=%s",
                    uploaded_file_name,
                    model_name,
                )
                if "LangChain dependencies are incompatible" in str(e):
                    st.error("Environment issue detected. Start Streamlit with the project venv interpreter.")
                    st.code(".\\.venv\\Scripts\\python.exe -m streamlit run app.py")
                else:
                    st.error("Analysis failed. Try a simpler question or switch to a smaller model.")
                with st.expander("Technical details"):
                    st.write(str(e))


def main():
    st.set_page_config(page_title=" AI Data Analyst", layout="wide")
    load_dotenv()

    with st.sidebar:
        st.title(" AI Analyst")
        st.markdown("Upload a CSV file and ask questions about your data.")

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            api_key = st.text_input("Enter Groq API Key:", type="password")

        model_name = st.selectbox(
            "Model",
            [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "gemma2-9b-it",
                "mixtral-8x7b-32768",
            ],
            help="Switch to a smaller model if you hit rate limits.",
        )

        st.markdown("### Safety")
        execution_confirmed = st.checkbox(
            "Enable advanced AI analysis",
            value=False,
            help="This runs AI-generated Python code for data analysis. Use only trusted CSV files.",
        )

    st.title(" Chat with your Data (CSV)")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is None:
        st.info("Upload a CSV file to start exploring Data Health, Root Cause, What-If, and AI analysis.")
        return

    initialize_file_session(uploaded_file.name)
    df = load_csv(uploaded_file)
    plot_path = get_plot_path()

    st.write("### Data Preview")
    st.dataframe(df.head())

    render_data_health_section(df)
    render_quick_insights_section(df)
    render_root_cause_section(df)
    render_what_if_section(df)
    render_analysis_journal_section()
    render_chat_section(df, api_key, model_name, execution_confirmed, uploaded_file.name, plot_path)


if __name__ == "__main__":
    main()
