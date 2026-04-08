import numpy as np
import pandas as pd


def aggregate_numeric(series, agg_method):
    if agg_method == "Mean":
        return float(series.mean())
    if agg_method == "Median":
        return float(series.median())
    return float(series.sum())


def compute_what_if_simulation(df, target_metric, driver_adjustments, agg_method):
    target_series = pd.to_numeric(df[target_metric], errors="coerce").dropna()
    if target_series.empty:
        return None

    baseline_target = aggregate_numeric(target_series, agg_method)
    impacts = []

    for driver_column, pct_change in driver_adjustments:
        driver_series = pd.to_numeric(df[driver_column], errors="coerce").dropna()
        if driver_series.empty:
            continue

        baseline_driver = aggregate_numeric(driver_series, agg_method)
        scenario_driver = baseline_driver * (1 + (pct_change / 100.0))
        delta_driver = scenario_driver - baseline_driver

        pair_df = df[[target_metric, driver_column]].copy()
        pair_df[target_metric] = pd.to_numeric(pair_df[target_metric], errors="coerce")
        pair_df[driver_column] = pd.to_numeric(pair_df[driver_column], errors="coerce")
        pair_df = pair_df.dropna()

        slope = 0.0
        r2 = 0.0
        if len(pair_df) >= 3:
            x = pair_df[driver_column].to_numpy(dtype=float)
            y = pair_df[target_metric].to_numpy(dtype=float)

            x_variance = float(np.var(x))
            if x_variance > 0:
                covariance = float(np.cov(x, y, ddof=0)[0, 1])
                slope = covariance / x_variance

                corr_matrix = np.corrcoef(x, y)
                correlation = float(corr_matrix[0, 1]) if corr_matrix.size >= 4 else 0.0
                if np.isfinite(correlation):
                    r2 = correlation ** 2

        predicted_delta = slope * delta_driver
        impacts.append(
            {
                "driver": driver_column,
                "baseline_driver": round(baseline_driver, 4),
                "scenario_change_pct": round(float(pct_change), 2),
                "scenario_driver": round(scenario_driver, 4),
                "slope_estimate": round(slope, 4),
                "relationship_strength_r2": round(r2, 4),
                "predicted_metric_impact": round(predicted_delta, 4),
            }
        )

    if not impacts:
        return None

    impact_df = pd.DataFrame(impacts)
    net_delta = float(impact_df["predicted_metric_impact"].sum())
    projected_target = baseline_target + net_delta
    projected_pct = None
    if baseline_target != 0:
        projected_pct = (net_delta / baseline_target) * 100

    return {
        "baseline_target": round(baseline_target, 4),
        "projected_target": round(projected_target, 4),
        "net_delta": round(net_delta, 4),
        "projected_pct": None if projected_pct is None else round(projected_pct, 2),
        "impact_df": impact_df,
    }
