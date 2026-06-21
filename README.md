# 🚦 EventFlow — Predictive Traffic Impact & Resource Allocation System

**Flipkart Gridlock Hackathon 2.0 — Round 2 (Prototype Phase)**
**Theme:** Event-Driven Congestion (Planned & Unplanned)

> How can historical and real-time data be used to forecast event-related
> traffic impact and recommend optimal manpower, barricading, and diversion plans?

---

## 🎯 Problem

Political rallies, festivals, sports events, construction activity, and sudden
gatherings create localized traffic breakdowns across Bengaluru. Today:

- Event impact is **not quantified in advance**
- Resource deployment is **experience-driven**, not data-driven
- There is **no post-event learning system** to improve future deployments

## 💡 Solution

EventFlow is a predictive system trained on the **real ASTraM (Bengaluru
Traffic Police) event dataset** — 8,173 anonymized incident records spanning
Nov 2023–Apr 2024 — with three modules:

| Module | What it does |
|---|---|
| **A. Impact Predictor** | Two LightGBM models predict (1) probability an event will be BTP-classified High priority, and (2) expected disruption duration, for any new event description across real historical monitoring junctions citywide |
| **B. Resource Recommendation Engine** | Rule-based layer converts model outputs into personnel counts, barricading decisions, and diversion requirements per junction |
| **C. Post-Event Learning Loop** *(roadmap)* | Actual outcomes feed back into the model, closing the loop the problem statement explicitly calls out as missing today |

This reuses feature-engineering techniques (spatial distance encoding,
cyclical time features, corridor/zone interactions) validated in Round 1 of
this hackathon, where the same approach scored **91.28** on the traffic
demand prediction leaderboard.

## 🏗️ Architecture

```
Event Input (cause, road closure, time, corridor)
        │
        ▼
┌────────────────────────┐
│   Feature Engineering   │  ← distance-to-center, cyclical hour/day,
│                         │     corridor/zone/event-cause encoding
└────────────┬─────────────┘
             ▼
┌─────────────────────────────────────┐
│  Priority Classifier (AUC 0.999)     │  ← P(High priority) per junction
│  Duration Regressor  (R² 0.41, log)  │  ← expected disruption length
└────────────┬─────────────────────────┘
             ▼
┌─────────────────────────┐
│  Recommendation Engine   │  ← severity band → personnel / barricade / divert
└────────────┬──────────────┘
             ▼
   Streamlit Dashboard (live map + prioritized action plan)
```

## 📊 Model Performance & Methodology Honesty

- **Priority model:** 5-fold CV AUC = **0.999**. A major real driver: incidents
  on named arterial corridors (Mysore Road, Bellary Road, ORR segments) are
  almost always BTP-tagged High priority — this is a genuine operational
  policy signal in the real data, not a modeling artifact. We verified this
  by checking class separability per corridor directly.
- **Duration model:** 5-fold CV R² = **0.41** (log scale). This is
  intentionally modest. Real incident-resolution timestamps are noisy — some
  "closed" tickets represent ongoing maintenance tracking (potholes, road
  conditions) left open for days, while active disruptions (breakdowns,
  accidents) typically resolve in under an hour. We did not inflate this
  number with leaky features; predictions are clipped to a realistic
  5–2000 minute range for operational use.
- **`requires_road_closure` was deliberately excluded** as a predictor of
  priority, since in BTP's real workflow it's decided *alongside* priority,
  not known in advance of it — including it would leak the decision itself.

An earlier iteration of this model used a self-constructed severity score
algebraically derived from priority + closure columns, which produced an
inflated R²=0.97 by reconstructing its own label formula rather than
predicting anything genuinely uncertain. We caught this, removed it, and
rebuilt against the real BTP-assigned `priority` field directly — we think
this kind of leakage discipline matters for a system meant to inform actual
police deployment decisions.

## 🚀 Running Locally

```bash
git clone <repo-url>
cd eventflow
pip install -r requirements.txt

# Place the real ASTraM dataset CSV at data/astram_raw.csv (already included)
python prepare_data.py     # clean + feature-engineer
python train_model.py      # train priority + duration models

streamlit run app.py
```

App opens at `http://localhost:8501`.

## 📁 Project Structure

```
eventflow/
├── app.py                    # Streamlit dashboard
├── prepare_data.py           # ASTraM data cleaning + feature engineering
├── train_model.py            # Model training (priority classifier + duration regressor)
├── recommendation_engine.py  # Risk → resource recommendation logic
├── requirements.txt
├── data/
│   ├── astram_raw.csv        # Real ASTraM event dataset (provided)
│   ├── astram_clean.csv      # Cleaned + feature-engineered version
│   ├── model.pkl             # Trained model bundle
│   └── encoders.pkl          # LabelEncoders for categorical features
└── README.md
```

## 🗺️ Case Study from the Real Data

The dataset literally contains two IPL match entries at M. Chinnaswamy
Stadium (`QueensStatueCircle` junction, CBD 2 corridor) — both correctly
tagged High priority by BTP. The same junction also recurs across the
dataset for accidents, potholes, vehicle breakdowns, and construction —
making it a genuine, data-identified chronic hotspot independent of any
single event. This is exactly the kind of cross-cause pattern detection
the hackathon problem statement asks for.

## 🔭 Roadmap Beyond Prototype

1. **Live CCTV integration** — real-time crowd/density estimation to refine
   predictions as an event unfolds, not just pre-event
2. **Post-event feedback loop** — log predicted vs. actual outcomes,
   retrain incrementally (directly addresses the "no learning system" gap)
3. **Route-level diversion planner** — integrate MapmyIndia road network
   data for specific alternate-route recommendations, not just zone flags
4. **Multi-event conflict detection** — flag overlapping impact windows
   (e.g., a concert and a cricket match the same evening, nearby corridors)

## 👤 Team

Shivam — Solo Builder, Round 1 Qualifier (Traffic Demand Prediction, Score: 91.28)

---

*Built for Flipkart × Bengaluru Traffic Police Gridlock Hackathon 2.0,
using the real ASTraM event dataset provided for Round 2.*
