import pandas as pd


def compute_data_health(df):
    row_count = len(df.index)
    col_count = len(df.columns)
    total_cells = max(row_count * col_count, 1)

    missing_by_col = df.isna().sum()
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

    duplicate_rows = int(df.duplicated().sum())

    type_issues = []
    for column in df.columns:
        series = df[column]
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
                non_null.str.contains(r"^\\d{1,4}[-/]\\d{1,2}[-/]\\d{1,4}", regex=True).mean()
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

    numeric_df = df.select_dtypes(include=["number"])
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
