"""
EventFlow — Resource Recommendation Engine
Converts model outputs (priority probability + predicted duration) into
actionable recommendations: personnel deployment, barricading, diversion.
Built around real BTP operational categories (High/Low priority).
"""

import pandas as pd


def severity_band(high_priority_proba: float) -> str:
    """Maps model's probability of High priority into an operational band."""
    if high_priority_proba >= 0.75:
        return "CRITICAL"
    elif high_priority_proba >= 0.5:
        return "HIGH"
    elif high_priority_proba >= 0.25:
        return "MODERATE"
    else:
        return "LOW"


def recommend_resources(high_priority_proba: float, duration_mins: float, is_planned: int):
    band = severity_band(high_priority_proba)

    base_personnel = {
        "CRITICAL": 8,
        "HIGH": 5,
        "MODERATE": 3,
        "LOW": 1,
    }[band]

    # Longer expected disruptions need sustained (not just initial) staffing
    duration_multiplier = 1.0
    if duration_mins > 180:
        duration_multiplier = 1.5
    elif duration_mins > 60:
        duration_multiplier = 1.2

    personnel = max(1, round(base_personnel * duration_multiplier))

    barricading = band in ["CRITICAL", "HIGH"]
    diversion_needed = band in ["CRITICAL", "HIGH"] or (band == "MODERATE" and duration_mins > 120)

    # Unplanned events need faster reactive deployment than planned ones
    response_window_mins = 15 if is_planned == 0 else 45

    return {
        "severity_band": band,
        "recommended_personnel": personnel,
        "barricading_required": barricading,
        "diversion_required": diversion_needed,
        "response_window_mins": response_window_mins,
    }


def generate_action_plan(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    predictions_df must contain: high_priority_proba, predicted_duration_mins,
    is_planned, plus identifying columns (e.g. junction/zone/lat/lon).
    Returns the same frame enriched with recommendation columns, sorted by
    descending priority probability.
    """
    plan = predictions_df.copy()
    plan["severity_band"] = plan["high_priority_proba"].apply(severity_band)

    recs = plan.apply(
        lambda r: recommend_resources(
            r["high_priority_proba"], r["predicted_duration_mins"], r["is_planned"]
        ),
        axis=1,
        result_type="expand",
    )
    plan = pd.concat(
        [plan, recs[["recommended_personnel", "barricading_required",
                      "diversion_required", "response_window_mins"]]],
        axis=1,
    )

    plan = plan.sort_values("high_priority_proba", ascending=False).reset_index(drop=True)
    return plan
