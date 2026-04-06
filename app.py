import streamlit as st
import pandas as pd
import os
import logging
import importlib
from pathlib import Path
from uuid import uuid4
from dotenv import load_dotenv


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
def compute_data_health(_df):
    row_count = len(_df.index)
    col_count = len(_df.columns)
    total_cells = max(row_count * col_count, 1)

    missing_by_col = _df.isna().sum()
    missing_cells = int(missing_by_col.sum())
    missing_columns = int((missing_by_col > 0).sum())
    missing_df = (
        missing_by_col[missing_by_col > 0]
        .sort_values(ascending=False)
        .to_frame("missing_count")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    if not missing_df.empty:
        missing_df["missing_pct_of_rows"] = (
            missing_df["missing_count"] / max(row_count, 1) * 100
        ).round(2)

    duplicate_rows = int(_df.duplicated().sum())

    type_issues = []
    for column in _df.columns:
        series = _df[column]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            non_null = series.dropna().astype(str).str.strip()
            if non_null.empty:
                continue

            numeric_candidate = pd.to_numeric(
                non_null.str.replace(",", "", regex=False),
                errors="coerce",
            )
            numeric_ratio = float(numeric_candidate.notna().mean())
            if numeric_ratio >= 0.85:
                type_issues.append(
                    {
                        "column": column,
                        "issue": "Likely numeric values stored as text",
                        "confidence_pct": round(numeric_ratio * 100, 2),
                        "suggested_fix": "Convert this column to numeric",
                    }
                )
                continue

            date_like_ratio = float(
                non_null.str.contains(r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}", regex=True).mean()
            )
            if date_like_ratio >= 0.5:
                try:
                    datetime_candidate = pd.to_datetime(non_null, errors="coerce", format="mixed")
                except TypeError:
                    datetime_candidate = pd.to_datetime(non_null, errors="coerce")

                datetime_ratio = float(datetime_candidate.notna().mean())
                if datetime_ratio >= 0.85:
                    type_issues.append(
                        {
                            "column": column,
                            "issue": "Likely date values stored as text",
                            "confidence_pct": round(datetime_ratio * 100, 2),
                            "suggested_fix": "Convert this column to datetime",
                        }
                    )

    type_issue_df = pd.DataFrame(type_issues)

    numeric_df = _df.select_dtypes(include=["number"])
    outlier_records = []
    outlier_cells = 0

    for column in numeric_df.columns:
        non_null = numeric_df[column].dropna()
        if len(non_null) < 4:
            continue

        q1 = non_null.quantile(0.25)
        q3 = non_null.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        mask = (non_null < lower_bound) | (non_null > upper_bound)
        outlier_count = int(mask.sum())

        if outlier_count > 0:
            outlier_cells += outlier_count
            outlier_records.append(
                {
                    "column": column,
                    "outlier_count": outlier_count,
                    "outlier_pct_of_rows": round(outlier_count / max(row_count, 1) * 100, 2),
                    "lower_bound": round(float(lower_bound), 4),
                    "upper_bound": round(float(upper_bound), 4),
                }
            )

    outlier_df = pd.DataFrame(outlier_records)
    if not outlier_df.empty:
        outlier_df = outlier_df.sort_values(by="outlier_count", ascending=False)

    numeric_non_null_cells = int(numeric_df.notna().sum().sum())
    missing_ratio = missing_cells / total_cells
    duplicate_ratio = duplicate_rows / max(row_count, 1)
    outlier_ratio = outlier_cells / max(numeric_non_null_cells, 1)

    penalty_missing = min(35.0, missing_ratio * 120)
    penalty_duplicates = min(25.0, duplicate_ratio * 200)
    penalty_outliers = min(20.0, outlier_ratio * 100)
    penalty_type_issues = min(20.0, len(type_issues) * 5.0)

    total_penalty = penalty_missing + penalty_duplicates + penalty_outliers + penalty_type_issues
    health_score = max(0.0, round(100.0 - total_penalty, 1))

    if health_score >= 90:
        health_status = "Excellent"
    elif health_score >= 75:
        health_status = "Good"
    elif health_score >= 60:
        health_status = "Needs Attention"
    else:
        health_status = "Poor"

    penalty_df = pd.DataFrame(
        [
            {"factor": "Missing values", "penalty": round(penalty_missing, 1)},
            {"factor": "Duplicate rows", "penalty": round(penalty_duplicates, 1)},
            {"factor": "Outlier values", "penalty": round(penalty_outliers, 1)},
            {"factor": "Type issues", "penalty": round(penalty_type_issues, 1)},
        ]
    )

    return {
        "health_score": health_score,
        "health_status": health_status,
        "rows": row_count,
        "columns": col_count,
        "missing_cells": missing_cells,
        "missing_columns": missing_columns,
        "missing_df": missing_df,
        "duplicate_rows": duplicate_rows,
        "type_issue_count": len(type_issues),
        "type_issue_df": type_issue_df,
        "outlier_cells": outlier_cells,
        "outlier_df": outlier_df,
        "penalty_df": penalty_df,
    }


@st.cache_data
def detect_date_columns(_df):
    candidate_columns = []
    sample_df = _df.head(2000)

    for column in sample_df.columns:
        series = sample_df[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            candidate_columns.append(column)
            continue

        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            continue

        non_null = series.dropna().astype(str).str.strip()
        if non_null.empty:
            continue

        try:
            parsed_dates = pd.to_datetime(non_null, errors="coerce", format="mixed")
        except TypeError:
            parsed_dates = pd.to_datetime(non_null, errors="coerce")

        if float(parsed_dates.notna().mean()) >= 0.70:
            candidate_columns.append(column)

    return candidate_columns


@st.cache_data
def get_period_labels(_df, date_column, period_grain):
    try:
        parsed_dates = pd.to_datetime(_df[date_column], errors="coerce", format="mixed")
    except TypeError:
        parsed_dates = pd.to_datetime(_df[date_column], errors="coerce")

    parsed_dates = parsed_dates.dropna()
    if parsed_dates.empty:
        return []

    if period_grain == "Daily":
        period_dt = parsed_dates.dt.floor("D")
        labels = period_dt.dt.strftime("%Y-%m-%d")
    elif period_grain == "Weekly":
        period_obj = parsed_dates.dt.to_period("W-SUN")
        period_dt = period_obj.dt.start_time
        labels = period_dt.dt.strftime("%Y-%m-%d")
    elif period_grain == "Quarterly":
        period_obj = parsed_dates.dt.to_period("Q")
        period_dt = period_obj.dt.to_timestamp()
        labels = period_obj.astype(str)
    else:  # Monthly
        period_obj = parsed_dates.dt.to_period("M")
        period_dt = period_obj.dt.to_timestamp()
        labels = period_obj.astype(str)

    ordered_periods = (
        pd.DataFrame({"period_dt": period_dt, "period_label": labels})
        .drop_duplicates(subset=["period_label"])
        .sort_values(by="period_dt")
    )

    return ordered_periods["period_label"].tolist()


@st.cache_data
def compute_root_cause_period(
    _df,
    metric_column,
    dimension_column,
    date_column,
    period_grain,
    baseline_period,
    comparison_period,
):
    working_df = _df[[metric_column, dimension_column, date_column]].copy()
    working_df["_metric"] = pd.to_numeric(working_df[metric_column], errors="coerce")

    try:
        parsed_dates = pd.to_datetime(working_df[date_column], errors="coerce", format="mixed")
    except TypeError:
        parsed_dates = pd.to_datetime(working_df[date_column], errors="coerce")

    working_df["_date"] = parsed_dates
    working_df["_dimension"] = working_df[dimension_column].fillna("Unknown").astype(str).str.strip()
    working_df.loc[working_df["_dimension"] == "", "_dimension"] = "Unknown"
    working_df = working_df.dropna(subset=["_metric", "_date"])

    if working_df.empty:
        return None

    if period_grain == "Daily":
        period_dt = working_df["_date"].dt.floor("D")
        period_label = period_dt.dt.strftime("%Y-%m-%d")
    elif period_grain == "Weekly":
        period_obj = working_df["_date"].dt.to_period("W-SUN")
        period_dt = period_obj.dt.start_time
        period_label = period_dt.dt.strftime("%Y-%m-%d")
    elif period_grain == "Quarterly":
        period_obj = working_df["_date"].dt.to_period("Q")
        period_dt = period_obj.dt.to_timestamp()
        period_label = period_obj.astype(str)
    else:  # Monthly
        period_obj = working_df["_date"].dt.to_period("M")
        period_dt = period_obj.dt.to_timestamp()
        period_label = period_obj.astype(str)

    working_df["_period"] = period_label

    scoped_df = working_df[working_df["_period"].isin([baseline_period, comparison_period])]
    if scoped_df.empty:
        return None

    baseline_df = scoped_df[scoped_df["_period"] == baseline_period]
    comparison_df = scoped_df[scoped_df["_period"] == comparison_period]

    baseline_group = baseline_df.groupby("_dimension")["_metric"].sum()
    comparison_group = comparison_df.groupby("_dimension")["_metric"].sum()

    contribution_df = (
        pd.concat(
            [
                baseline_group.rename("baseline_value"),
                comparison_group.rename("comparison_value"),
            ],
            axis=1,
        )
        .fillna(0)
        .reset_index()
        .rename(columns={"_dimension": "dimension"})
    )

    contribution_df["change"] = contribution_df["comparison_value"] - contribution_df["baseline_value"]
    contribution_df["abs_change"] = contribution_df["change"].abs()

    total_change = float(contribution_df["change"].sum())
    total_abs_change = float(contribution_df["abs_change"].sum())

    if total_change != 0:
        contribution_df["contribution_pct"] = (
            contribution_df["change"] / total_change * 100
        )
    else:
        contribution_df["contribution_pct"] = 0.0

    if total_abs_change > 0:
        contribution_df["share_of_abs_change_pct"] = (
            contribution_df["abs_change"] / total_abs_change * 100
        )
    else:
        contribution_df["share_of_abs_change_pct"] = 0.0

    contribution_df = contribution_df.sort_values(by="abs_change", ascending=False)

    baseline_total = float(baseline_df["_metric"].sum())
    comparison_total = float(comparison_df["_metric"].sum())
    net_change = comparison_total - baseline_total
    pct_change = None
    if baseline_total != 0:
        pct_change = (net_change / baseline_total) * 100

    numeric_columns = [
        "baseline_value",
        "comparison_value",
        "change",
        "abs_change",
        "contribution_pct",
        "share_of_abs_change_pct",
    ]
    contribution_df[numeric_columns] = contribution_df[numeric_columns].round(2)

    return {
        "baseline_total": round(baseline_total, 2),
        "comparison_total": round(comparison_total, 2),
        "net_change": round(net_change, 2),
        "pct_change": None if pct_change is None else round(pct_change, 2),
        "contribution_df": contribution_df,
    }


def is_query_safe(user_question):
    lowered = user_question.lower()
    for pattern in BLOCKED_QUERY_PATTERNS:
        if pattern in lowered:
            return False
    return True


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
        temperature=0
    )
    return create_pandas_dataframe_agent(
        llm,
        _df,
        verbose=False,
        allow_dangerous_code=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=15,
    )


# 1. Config & Setup
st.set_page_config(page_title=" AI Data Analyst", layout="wide")
load_dotenv()

# Sidebar for API Key (Optional: allows users to use their own key)
with st.sidebar:
    st.title(" AI Analyst")
    st.markdown("Upload a CSV file and ask questions about your data.")
    
    # Check for env key, otherwise ask user
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        api_key = st.text_input("Enter Groq API Key:", type="password")

    model_name = st.selectbox("Model", [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "gemma2-9b-it",
        "mixtral-8x7b-32768",
    ], help="Switch to a smaller model if you hit rate limits.")

    st.markdown("### Safety")
    execution_confirmed = st.checkbox(
        "Enable advanced AI analysis",
        value=False,
        help="This runs AI-generated Python code for data analysis. Use only trusted CSV files.",
    )

# 2. Main UI
st.title(" Chat with your Data (CSV)")

# File Uploader
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Detect file change and clear chat history
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.messages = []
        st.session_state.pop("session_id", None)

    df = load_csv(uploaded_file)
    plot_path = get_plot_path()
    
    # Show Data Preview
    st.write("### Data Preview")
    st.dataframe(df.head())

    # Day 2 - Data Health Check workflow
    st.write("### Data Health Check")
    health = compute_data_health(df)

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

    # Quick Insights
    st.write("### Quick Insights")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Dataset Summary"):
            st.write(df.describe())

    with col2:
        if st.button("Correlation Matrix"):
            st.write(df.corr(numeric_only=True))

    # Day 3 - Root Cause Explorer workflow
    st.write("### Root Cause Explorer")
    st.caption("Explain what drove metric change between two time periods.")

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    date_columns = detect_date_columns(df)

    if not numeric_columns:
        st.info("Root Cause Explorer needs at least one numeric metric column.")
    elif not date_columns:
        st.info("No date-like column detected. Add a date column to run period-over-period root cause analysis.")
    else:
        root_metric = st.selectbox("Metric to explain", numeric_columns, key="root_metric")

        dimension_candidates = [
            column
            for column in df.columns
            if column != root_metric
            and column not in date_columns
            and (
                pd.api.types.is_object_dtype(df[column])
                or pd.api.types.is_string_dtype(df[column])
                or pd.api.types.is_categorical_dtype(df[column])
                or pd.api.types.is_bool_dtype(df[column])
                or (
                    pd.api.types.is_integer_dtype(df[column])
                    and 2 <= df[column].nunique(dropna=True) <= 40
                )
            )
        ]

        if not dimension_candidates:
            st.info("Root Cause Explorer needs at least one categorical column for breakdown.")
        else:
            root_dimension = st.selectbox("Break down change by", dimension_candidates, key="root_dimension")
            root_date = st.selectbox("Date column", date_columns, key="root_date")
            period_grain = st.selectbox(
                "Period grain",
                ["Monthly", "Quarterly", "Weekly", "Daily"],
                key="root_period_grain",
            )

            period_labels = get_period_labels(df, root_date, period_grain)

            if len(period_labels) < 2:
                st.warning("Need at least two distinct periods to compare.")
            else:
                default_comparison_index = len(period_labels) - 1
                default_baseline_index = max(0, default_comparison_index - 1)

                baseline_period = st.selectbox(
                    "Baseline period",
                    period_labels,
                    index=default_baseline_index,
                    key="root_baseline_period",
                )

                comparison_candidates = [
                    period for period in period_labels if period != baseline_period
                ]
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

                root_result = compute_root_cause_period(
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
                else:
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
                    st.bar_chart(
                        top_driver_df.set_index("dimension")[["change"]],
                        height=320,
                    )

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

    if not api_key:
        st.info("Add your API key in the sidebar to enable AI chat analysis.")
    else:
        if not execution_confirmed:
            st.info("Enable advanced AI analysis in the sidebar to run custom questions and chart generation.")

        # 4. Chat Interface
        st.write("### Ask a Question")
        user_question = st.text_input("Example: 'Plot a bar chart of Sales by Region and save it as plot.png'")

        if st.button("Analyze"):
            if not execution_confirmed:
                st.warning("Enable advanced AI analysis in the sidebar before running a question.")
            elif not user_question.strip():
                st.warning("Please enter a question first.")
            elif not is_query_safe(user_question):
                st.error("This question looks unsafe. Please remove system or file commands and try again.")
            else:
                with st.spinner("Analyzing data..."):
                    try:
                        # Setup the agent only when analysis is explicitly enabled.
                        agent = create_agent(df, api_key, uploaded_file.name, model_name)

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
                            uploaded_file.name,
                            model_name,
                            user_question[:200],
                        )

                        if hasattr(agent, "invoke"):
                            response_payload = agent.invoke({"input": enhanced_query})
                            if isinstance(response_payload, dict):
                                response = response_payload.get("output") or str(response_payload)
                            else:
                                response = str(response_payload)
                        else:
                            response = agent.run(enhanced_query)

                        st.success("Analysis Complete!")
                        st.write(response)

                        if plot_path.exists():
                            st.image(str(plot_path), caption="Generated Visualization")

                    except Exception as e:
                        logger.exception(
                            "Analyze failed | file=%s | model=%s",
                            uploaded_file.name,
                            model_name,
                        )
                        if "LangChain dependencies are incompatible" in str(e):
                            st.error("Environment issue detected. Start Streamlit with the project venv interpreter.")
                            st.code(".\\.venv\\Scripts\\python.exe -m streamlit run app.py")
                        else:
                            st.error("Analysis failed. Try a simpler question or switch to a smaller model.")
                        with st.expander("Technical details"):
                            st.write(str(e))
else:
    st.info("Upload a CSV file to start exploring Data Health and AI analysis.")