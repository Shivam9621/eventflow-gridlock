"""
EventFlow — Model Training (Real ASTraM Dataset)
Trains two models on genuinely independent targets:
  1. Priority Classifier — predicts BTP's actual High/Low priority label
     (real ground truth assigned by traffic police operators, not derived)
  2. Duration Regressor — predicts expected disruption duration (minutes)
Both driven by spatial (lat/lon/zone/corridor), temporal, and event-type
features — the SAME feature engineering philosophy validated in Round 1
(geohash/time interactions, R²=0.91 there), now applied to real BTP data.

NOTE ON METHODOLOGY: an earlier version of this model trained against a
self-constructed "severity_score" that was algebraically derived from the
priority + road-closure columns themselves — this produced an inflated
R²=0.97 that was reconstructing a formula, not predicting anything
genuinely uncertain. This version instead predicts the real BTP-assigned
`priority` label directly via classification, which is the actual
decision signal an officer would want forecasted before an event happens.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

CLEAN_PATH = "/home/claude/eventflow/data/astram_clean.csv"
MODEL_PATH = "/home/claude/eventflow/data/model.pkl"
ENCODERS_PATH = "/home/claude/eventflow/data/encoders.pkl"

# Bengaluru center for distance-from-center feature
CITY_LAT, CITY_LON = 12.9716, 77.5946


def add_features(df):
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["dist_center"] = np.sqrt((df["latitude"] - CITY_LAT)**2 + (df["longitude"] - CITY_LON)**2)
    df["requires_road_closure"] = df["requires_road_closure"].astype(str).str.upper().eq("TRUE").astype(int)
    df["is_planned"] = (df["event_type"] == "planned").astype(int)
    return df


def train_priority_model(df, features):
    """Predicts BTP's real priority label (High=1 / Low=0). Road closure is
    EXCLUDED from features here since it's operationally decided alongside
    priority, not known beforehand — using it would leak the decision itself."""
    X = df[features]
    y = (df["priority"] == "High").astype(int)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba = np.zeros(len(df))

    params = {
        "objective": "binary", "metric": "auc",
        "n_estimators": 1200, "learning_rate": 0.03, "num_leaves": 63,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "random_state": 42, "verbose": -1,
    }

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(60, verbose=False)])
        oof_proba[val_idx] = m.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y, oof_proba)
    acc = accuracy_score(y, (oof_proba > 0.5).astype(int))
    print(f"Priority Model — OOF AUC = {auc:.4f}, Accuracy = {acc:.4f}")

    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(X, y)
    return final_model, auc


def train_duration_model(df, features):
    # Only rows with known, sane duration; clip extreme tail for stability
    df_dur = df[df["duration_mins"].notna()].copy()
    cap = df_dur["duration_mins"].quantile(0.95)
    df_dur = df_dur[df_dur["duration_mins"] <= cap]
    df_dur["log_duration"] = np.log1p(df_dur["duration_mins"])

    X = df_dur[features]
    y = df_dur["log_duration"]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(df_dur))

    params = {
        "objective": "regression", "metric": "rmse",
        "n_estimators": 1000, "learning_rate": 0.03, "num_leaves": 31,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "random_state": 42, "verbose": -1,
    }

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(60, verbose=False)])
        oof[val_idx] = m.predict(X_val)

    r2 = r2_score(y, oof)
    mae_mins = mean_absolute_error(np.expm1(y), np.expm1(oof))
    print(f"Duration Model — OOF R² (log-scale) = {r2:.4f}, MAE = {mae_mins:.1f} mins")

    final_model = lgb.LGBMRegressor(**params)
    final_model.fit(X, y)
    return final_model, r2


def main():
    df = pd.read_csv(CLEAN_PATH)
    df = add_features(df)

    cat_cols = ["event_cause", "zone", "corridor", "veh_type"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # NOTE: requires_road_closure excluded from priority-model features
    # (see docstring) but kept available for the duration model and for
    # display purposes downstream.
    priority_features = [
        "latitude", "longitude", "dist_center", "event_cause", "zone",
        "corridor", "veh_type", "is_planned",
        "hour", "dayofweek", "is_weekend", "is_peak_hour", "month",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ]
    duration_features = priority_features + ["requires_road_closure"]

    print("=" * 60)
    priority_model, pri_auc = train_priority_model(df, priority_features)
    print("=" * 60)
    duration_model, dur_r2 = train_duration_model(df, duration_features)
    print("=" * 60)

    bundle = {
        "priority_model": priority_model,
        "duration_model": duration_model,
        "priority_features": priority_features,
        "duration_features": duration_features,
        "priority_auc": pri_auc,
        "duration_r2": dur_r2,
    }
    joblib.dump(bundle, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    print(f"\n✅ Models saved to {MODEL_PATH}")
    print(f"✅ Encoders saved to {ENCODERS_PATH}")

    imp = pd.DataFrame({
        "feature": priority_features,
        "importance": priority_model.feature_importances_
    }).sort_values("importance", ascending=False)
    print("\nTop features (priority model):")
    print(imp.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

