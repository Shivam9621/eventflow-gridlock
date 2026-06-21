"""
EventFlow — Data Preparation (Real ASTraM Dataset)
Cleans and engineers features from the real Bengaluru Traffic Police (ASTraM)
event dataset for congestion severity + duration prediction.
"""

import pandas as pd
import numpy as np

RAW_PATH = "/home/claude/eventflow/data/astram_raw.csv"
CLEAN_PATH = "/home/claude/eventflow/data/astram_clean.csv"


def load_and_clean():
    df = pd.read_csv(RAW_PATH)

    # ── Parse datetimes ──────────────────────────────────────────
    for col in ["start_datetime", "end_datetime", "closed_datetime", "resolved_datetime"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # ── Filter to valid coordinates within Bengaluru bounds ──────
    df = df[
        (df["latitude"].between(12.7, 13.3)) &
        (df["longitude"].between(77.2, 77.9))
    ].copy()

    # ── Duration target: time from start to closed/resolved ─────
    df["end_time"] = df["closed_datetime"].fillna(df["resolved_datetime"])
    df["duration_mins"] = (df["end_time"] - df["start_datetime"]).dt.total_seconds() / 60

    # Clean unrealistic durations (negative, or absurdly long > 30 days)
    df.loc[(df["duration_mins"] < 0) | (df["duration_mins"] > 30 * 24 * 60), "duration_mins"] = np.nan

    # ── Standardize event_cause (fix casing inconsistency) ───────
    df["event_cause"] = df["event_cause"].astype(str).str.strip().str.lower()
    df["event_cause"] = df["event_cause"].replace({"debris": "debris", "fog / low visibility": "fog_low_visibility"})

    # ── Time features ────────────────────────────────────────────
    df["hour"] = df["start_datetime"].dt.hour
    df["dayofweek"] = df["start_datetime"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["is_peak_hour"] = df["hour"].isin([8, 9, 10, 18, 19, 20, 21]).astype(int)
    df["month"] = df["start_datetime"].dt.month

    # ── Target: severity score (0-100) derived from priority + road closure ──
    # Real BTP-assigned priority is our ground-truth signal; road closure adds weight
    priority_base = df["priority"].map({"High": 65, "Low": 30}).fillna(30)
    closure_bonus = df["requires_road_closure"].astype(str).str.upper().eq("TRUE").astype(int) * 20
    status_bonus = df["status"].map({"active": 10, "closed": 0, "resolved": -5}).fillna(0)

    df["severity_score"] = (priority_base + closure_bonus + status_bonus).clip(0, 100)

    # ── Fill required fields ─────────────────────────────────────
    df["zone"] = df["zone"].fillna("Unknown_Zone")
    df["corridor"] = df["corridor"].fillna("Non-corridor")
    df["junction"] = df["junction"].fillna("Unknown_Junction")
    df["veh_type"] = df["veh_type"].fillna("not_applicable")

    keep_cols = [
        "id", "event_type", "event_cause", "latitude", "longitude",
        "requires_road_closure", "priority", "status", "zone", "corridor",
        "junction", "veh_type", "hour", "dayofweek", "is_weekend",
        "is_peak_hour", "month", "duration_mins", "severity_score",
        "address", "start_datetime",
    ]
    df_clean = df[keep_cols].copy()

    return df_clean


if __name__ == "__main__":
    df = load_and_clean()
    print("Cleaned shape:", df.shape)
    print("\nSeverity score distribution:\n", df["severity_score"].describe())
    print("\nDuration (mins) — non-null:", df["duration_mins"].notna().sum())
    print(df["duration_mins"].describe())
    print("\nevent_cause distribution:\n", df["event_cause"].value_counts())
    print("\nzone distribution:\n", df["zone"].value_counts())

    df.to_csv(CLEAN_PATH, index=False)
    print(f"\n✅ Saved cleaned data to {CLEAN_PATH}")
