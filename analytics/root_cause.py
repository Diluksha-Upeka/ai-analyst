import pandas as pd


def detect_date_columns(df):
    candidate_columns = []
    sample_df = df.head(2000)

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


def get_period_labels(df, date_column, period_grain):
    try:
        parsed_dates = pd.to_datetime(df[date_column], errors="coerce", format="mixed")
    except TypeError:
        parsed_dates = pd.to_datetime(df[date_column], errors="coerce")

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


def compute_root_cause_period(
    df,
    metric_column,
    dimension_column,
    date_column,
    period_grain,
    baseline_period,
    comparison_period,
):
    working_df = df[[metric_column, dimension_column, date_column]].copy()
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
