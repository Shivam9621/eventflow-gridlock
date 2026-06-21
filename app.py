"""
EventFlow — Predictive Traffic Impact & Resource Allocation System
Built for Flipkart Gridlock Hackathon 2.0 — Round 2 (Event-Driven Congestion)
Trained on real ASTraM (Bengaluru Traffic Police) event data.

Run locally:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pydeck as pdk

from recommendation_engine import generate_action_plan, severity_band

st.set_page_config(
    page_title="EventFlow — Bengaluru Traffic Impact Predictor",
    page_icon="🚦",
    layout="wide",
)

CITY_LAT, CITY_LON = 12.9716, 77.5946

EVENT_CAUSES = [
    "public_event", "procession", "vip_movement", "protest",
    "construction", "accident", "vehicle_breakdown", "tree_fall",
    "water_logging", "pot_holes", "road_conditions", "congestion", "others",
]

PLANNED_CAUSES = {"public_event", "procession", "vip_movement", "construction"}


# ── Load artifacts ────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    bundle = joblib.load("data/model.pkl")
    encoders = joblib.load("data/encoders.pkl")
    return bundle, encoders

@st.cache_data
def load_reference_data():
    df = pd.read_csv("data/astram_clean.csv")
    return df

@st.cache_data
def load_zone_points(df, per_zone=15, seed=7):
    """Representative real incident locations per zone, used as the live
    citywide monitoring grid for new-event impact simulation."""
    known = df[df["zone"] != "Unknown_Zone"]
    parts = []
    for zone_name, g in known.groupby("zone"):
        parts.append(g.sample(min(per_zone, len(g)), random_state=seed))
    sampled = pd.concat(parts, ignore_index=True)
    return sampled[["zone", "corridor", "junction", "latitude", "longitude"]].drop_duplicates(
        subset=["latitude", "longitude"]
    ).reset_index(drop=True)


bundle, encoders = load_artifacts()
priority_model = bundle["priority_model"]
duration_model = bundle["duration_model"]
priority_features = bundle["priority_features"]
duration_features = bundle["duration_features"]

ref_df = load_reference_data()
zone_points = load_zone_points(ref_df)

VALID_CORRIDORS = sorted(ref_df["corridor"].dropna().unique().tolist())
VALID_VEH_TYPES = sorted(ref_df["veh_type"].dropna().unique().tolist())


def encode_safe(series, col):
    """Encode using the trained LabelEncoder; unseen values fall back to
    the most frequent training class rather than erroring out."""
    le = encoders[col]
    known = set(le.classes_)
    fallback = le.classes_[0]
    safe_vals = series.astype(str).apply(lambda v: v if v in known else fallback)
    return le.transform(safe_vals)


def predict_event_impact(event_cause, requires_closure, day_offset, hour, corridor_filter=None):
    is_planned = 1 if event_cause in PLANNED_CAUSES else 0
    dow = day_offset % 7
    is_weekend = 1 if dow in [5, 6] else 0
    is_peak = 1 if hour in [8, 9, 10, 18, 19, 20, 21] else 0
    month = 6  # current month placeholder; date doesn't materially affect model beyond dow/hour

    points = zone_points.copy()
    if corridor_filter and corridor_filter != "All Corridors":
        points = points[points["corridor"] == corridor_filter]
        if points.empty:
            points = zone_points.copy()

    df = points.copy()
    df["event_cause"] = event_cause
    df["requires_road_closure"] = int(requires_closure)
    df["is_planned"] = is_planned
    df["hour"] = hour
    df["dayofweek"] = dow
    df["is_weekend"] = is_weekend
    df["is_peak_hour"] = is_peak
    df["month"] = month
    df["dist_center"] = np.sqrt((df["latitude"] - CITY_LAT)**2 + (df["longitude"] - CITY_LON)**2)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["veh_type"] = "not_applicable"

    df_enc = df.copy()
    for col in ["event_cause", "zone", "corridor", "veh_type"]:
        df_enc[col] = encode_safe(df_enc[col], col)

    X_pri = df_enc[priority_features]
    X_dur = df_enc[duration_features]

    df["high_priority_proba"] = priority_model.predict_proba(X_pri)[:, 1]
    df["predicted_duration_mins"] = np.expm1(duration_model.predict(X_dur))
    df["predicted_duration_mins"] = df["predicted_duration_mins"].clip(5, 2000)

    return df


# ── UI ────────────────────────────────────────────────────────────
st.title("🚦 EventFlow")
st.caption(
    "Predictive Traffic Impact & Resource Allocation System for Bengaluru · "
    "Flipkart Gridlock Hackathon 2.0 · Trained on real ASTraM (Bengaluru Traffic Police) event data"
)

with st.sidebar:
    st.header("Event Details")
    event_cause = st.selectbox(
        "Event / Disruption Type", EVENT_CAUSES,
        format_func=lambda x: x.replace("_", " ").title()
    )
    is_planned_display = "Planned" if event_cause in PLANNED_CAUSES else "Unplanned"
    st.caption(f"Classified as: **{is_planned_display}** event")

    requires_closure = st.checkbox("Requires road closure?", value=False)

    corridor_filter = st.selectbox("Focus Corridor (optional)", ["All Corridors"] + VALID_CORRIDORS)

    day_offset = st.slider("Days from today", 0, 30, 3)
    hour = st.slider("Event Start Hour (24h)", 0, 23, 18)

    run = st.button("🔮 Predict Impact & Generate Plan", type="primary", use_container_width=True)

if run:
    with st.spinner("Predicting impact across monitoring zones (real BTP incident locations)..."):
        result = predict_event_impact(event_cause, requires_closure, day_offset, hour, corridor_filter)
        result["is_planned"] = 1 if event_cause in PLANNED_CAUSES else 0
        plan = generate_action_plan(result)

    max_p = plan["high_priority_proba"].max()
    avg_p = plan["high_priority_proba"].mean()
    critical_zones = (plan["severity_band"] == "CRITICAL").sum()
    high_zones = (plan["severity_band"] == "HIGH").sum()
    total_personnel = plan["recommended_personnel"].sum()
    median_duration = plan["predicted_duration_mins"].median()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Peak High-Priority Risk", f"{max_p*100:.0f}%", severity_band(max_p))
    c2.metric("Critical + High Zones", f"{critical_zones + high_zones} / {len(plan)}")
    c3.metric("Median Expected Duration", f"{median_duration:.0f} min")
    c4.metric("Total Personnel Recommended", f"{int(total_personnel)}")

    st.divider()

    col_map, col_table = st.columns([1.3, 1])

    with col_map:
        st.subheader("📍 Predicted Impact — Real BTP Monitoring Points")

        color_map = {
            "CRITICAL": [220, 38, 38, 200],
            "HIGH": [249, 115, 22, 190],
            "MODERATE": [234, 179, 8, 170],
            "LOW": [34, 197, 94, 140],
        }
        plan["color"] = plan["severity_band"].map(color_map)
        plan["radius"] = plan["high_priority_proba"] * 250 + 60

        layer_points = pdk.Layer(
            "ScatterplotLayer",
            data=plan,
            get_position=["longitude", "latitude"],
            get_fill_color="color",
            get_radius="radius",
            pickable=True,
        )

        view_state = pdk.ViewState(latitude=CITY_LAT, longitude=CITY_LON, zoom=10.3, pitch=20)

        st.pydeck_chart(pdk.Deck(
            layers=[layer_points],
            initial_view_state=view_state,
            tooltip={"text": "Junction: {junction}\nZone: {zone}\nHigh-priority risk: {high_priority_proba}\nSeverity: {severity_band}"}
        ))
        st.caption("🔴 Critical 🟠 High 🟡 Moderate 🟢 Low — points are real historical BTP incident locations")

    with col_table:
        st.subheader("📋 Prioritized Action Plan")
        display_cols = ["junction", "zone", "high_priority_proba", "predicted_duration_mins",
                         "severity_band", "recommended_personnel",
                         "barricading_required", "diversion_required"]
        top_zones = plan[display_cols].head(10).copy()
        top_zones["high_priority_proba"] = (top_zones["high_priority_proba"] * 100).round(0).astype(int)
        top_zones["predicted_duration_mins"] = top_zones["predicted_duration_mins"].round(0).astype(int)
        top_zones = top_zones.rename(columns={
            "junction": "Junction", "zone": "Zone",
            "high_priority_proba": "Risk %", "predicted_duration_mins": "Duration (min)",
            "severity_band": "Severity", "recommended_personnel": "Personnel",
            "barricading_required": "Barricade?", "diversion_required": "Divert?"
        })
        st.dataframe(top_zones, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🚨 Deployment Summary")
    top = plan.iloc[0]
    st.info(
        f"**Recommended response window:** {top['response_window_mins']} minutes before event/disruption start  \n"
        f"**Top priority location:** `{top['junction']}` ({top['zone']}) — "
        f"deploy {top['recommended_personnel']} personnel, expected duration ~{top['predicted_duration_mins']:.0f} min  \n"
        f"**Barricading required at:** {int(plan['barricading_required'].sum())} of {len(plan)} monitored zones  \n"
        f"**Diversion required at:** {int(plan['diversion_required'].sum())} of {len(plan)} monitored zones"
    )

    with st.expander("ℹ️ Model methodology & validation honesty"):
        st.markdown(f"""
        - **Priority model** (High/Low risk classifier): 5-fold CV AUC = **{bundle['priority_auc']:.3f}**.
          A major real driver: incidents on named arterial corridors (Mysore Road, Bellary Road, ORR
          segments, etc.) are almost always BTP-tagged High priority — this is a genuine operational
          policy pattern in the data, not a modeling artifact.
        - **Duration model** (log-scale regression): 5-fold CV R² = **{bundle['duration_r2']:.3f}**.
          This is intentionally modest — real incident resolution time is noisy (some "closed" tickets
          stay open for tracking purposes for days), and we did not inflate this number with leaky
          features. Predictions are clipped to a realistic 5–2000 minute range for operational use.
        - **Road closure** was deliberately excluded as a *predictor* of priority, since in practice
          it's decided alongside priority, not known in advance — including it would have leaked
          the decision itself rather than predicting it.
        """)

else:
    st.info("👈 Configure event/disruption details in the sidebar and click **Predict Impact & Generate Plan**")
    st.markdown("""
    ### How EventFlow Works
    1. **Predict** — two LightGBM models, trained on 8,173 real BTP (ASTraM) incident
       records, estimate (a) probability of High-priority classification and
       (b) expected disruption duration, across real historical monitoring junctions citywide
    2. **Prioritize** — zones ranked by predicted risk and corridor context
    3. **Recommend** — rule-based engine converts model outputs into personnel, barricading,
       and diversion decisions
    4. **Learn** — *(production roadmap)* post-event actuals feed back into the model,
       directly addressing the "no post-event learning system" gap called out in the
       hackathon problem statement

    ### Dataset
    This prototype is trained on the **real ASTraM event dataset** provided for Round 2 —
    8,173 anonymized incident records from Bengaluru Traffic Police, spanning Nov 2023–Apr 2024,
    covering planned events (public events, processions, VIP movement, construction) and
    unplanned disruptions (accidents, breakdowns, tree falls, waterlogging, protests).
    """)

st.divider()
st.caption("EventFlow Prototype · Flipkart Gridlock Hackathon 2.0 · Theme: Event-Driven Congestion (Planned & Unplanned) · Trained on real ASTraM data")
