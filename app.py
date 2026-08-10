"""
Apprentice risk dashboard

Reads raw apprentice feature data, scores it using calculate_risk_score.py
(the single source of truth for weights/caps), and shows a filterable,
sortable table with a drill-down into why any individual apprentice was
flagged.

Run with: streamlit run app.py
"""

import pandas as pd
import streamlit as st

from calculate_risk_score import calculate_risk_score, WEIGHTS, CAPS, FS_STATUS_CONCERN

DATA_PATH = "synthetic_apprentice_risk_data.csv"

st.set_page_config(page_title="Apprentice Risk Dashboard", layout="wide")


@st.cache_data
def load_and_score(path):
    df = pd.read_csv(path)
    return calculate_risk_score(df)


def top_factors_for_row(row, top_n=3):
    """Returns the top_n factors driving this apprentice's score, as a
    list of (factor_label, contribution) tuples, sorted highest first."""
    components = {
        "Attendance": (100 - row["attendance_pct"]) / 100,
        "OTJ deficit": max(row["otj_deficit_hours"], 0) / CAPS["otj_deficit_hours"],
        "Assignments outstanding": row["assignments_outstanding_count"] / CAPS["assignments_outstanding_count"],
        "Assignments late": row["assignments_late_count"] / CAPS["assignments_late_count"],
        "Days since OneFile update": row["days_since_onefile_update"] / CAPS["days_since_onefile_update"],
        "Days since last mentoring": row["days_since_last_mentoring"] / CAPS["days_since_last_mentoring"],
        "Days since last progress review": row["days_since_last_progress_review"] / CAPS["days_since_last_progress_review"],
        "FS Maths status": FS_STATUS_CONCERN[row["fs_maths_status"]],
        "FS English status": FS_STATUS_CONCERN[row["fs_english_status"]],
    }
    weight_keys = list(WEIGHTS.keys())
    label_to_key = dict(zip(components.keys(), weight_keys))
    contributions = {
        label: WEIGHTS[label_to_key[label]] * min(max(value, 0), 1)
        for label, value in components.items()
    }
    ranked = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


st.title("Apprentice Risk Dashboard")
st.caption("Proof of concept with synthetic apprentice data. Risk score is a weighted heuristic.")

try:
    scored = load_and_score(DATA_PATH)
except FileNotFoundError:
    st.error(f"Couldn't find {DATA_PATH}. Make sure it's in the same folder as this app.")
    st.stop()

# --- Filters ---
col1, col2 = st.columns(2)
with col1:
    cohorts = sorted(scored["cohort"].unique())
    selected_cohorts = st.multiselect("Cohort", cohorts, default=[])
with col2:
    band_filter = st.selectbox("Risk band", ["All", "High only", "Medium and above"])

filtered = scored.copy()
if selected_cohorts:
    filtered = filtered[filtered["cohort"].isin(selected_cohorts)]
if band_filter == "High only":
    filtered = filtered[filtered["risk_band"] == "High"]
elif band_filter == "Medium and above":
    filtered = filtered[filtered["risk_band"].isin(["Medium", "High"])]

# --- Summary metrics ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total apprentices", len(filtered))
m2.metric("High risk", (filtered["risk_band"] == "High").sum())
m3.metric("Medium risk", (filtered["risk_band"] == "Medium").sum())
m4.metric("Low risk", (filtered["risk_band"] == "Low").sum())

st.divider()

# --- Table ---
display_cols = [
    "synthetic_apprentice_id",
    "cohort",
    "attendance_pct",
    "otj_deficit_hours",
    "days_since_last_mentoring",
    "assignments_outstanding_count",
    "assignments_late_count",
    "risk_score",
    "risk_band",
]

st.dataframe(
    filtered[display_cols].sort_values("risk_score", ascending=False),
    width="stretch",
    hide_index=True,
)

st.divider()

# --- Drill-down ---
st.subheader("Why was someone flagged?")
apprentice_options = filtered.sort_values("risk_score", ascending=False)["synthetic_apprentice_id"].tolist()

if not apprentice_options:
    st.info("No apprentices match the current filters.")
else:
    selected_id = st.selectbox("Select an apprentice", apprentice_options)
    row = filtered[filtered["synthetic_apprentice_id"] == selected_id].iloc[0]

    left, right = st.columns([1, 2])
    with left:
        st.metric("Risk score", f"{row['risk_score']:.2f}")
        st.metric("Risk band", row["risk_band"])
        st.metric("Cohort", row["cohort"])

    with right:
        st.markdown("**Top contributing factors:**")
        top_factors = top_factors_for_row(row, top_n=3)
        for label, contribution in top_factors:
            st.write(f"- {label} (contributes {contribution:.2f} to the score)")

    with st.expander("Full raw data for this apprentice"):
        st.json(row.drop(labels=["risk_score", "risk_band"]).to_dict())

# --- Download ---
st.divider()
csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered results as CSV", csv_bytes, "filtered_risk_results.csv", "text/csv")
