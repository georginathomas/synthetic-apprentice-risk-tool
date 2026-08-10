"""
Apprentice risk scorer

Applies a weighted scoring formula to the raw synthetic feature
CSV.

Weights below come from a decided rank ordering (1 = most important) so the total sums to 1.
"""

import numpy as np
import pandas as pd

WEIGHTS = {
    "otj_deficit_hours": 0.350,
    "assignments_outstanding_count": 0.190,
    "assignments_late_count": 0.190,
    "attendance_pct": 0.098,
    "days_since_onefile_update": 0.061,
    "days_since_last_mentoring": 0.049,
    "days_since_last_progress_review": 0.037,
    "fs_english_status": 0.012,
    "fs_maths_status": 0.012,
}

# Caps used to scale each raw value to a 0-1 "concern" range before
# weighting.
CAPS = {
    "otj_deficit_hours": 30,       # 30+ hours deficit = max concern
    "assignments_outstanding_count": 5,
    "assignments_late_count": 5,
    "days_since_onefile_update": 30,      # a month with no timesheet update = max concern
    "days_since_last_mentoring": 60,       # two months with no mentoring update = max concern
    "days_since_last_progress_review": 90, # progress review overdue = max concern
}

# FS status -> concern value. Flat mapping for now: achieved and
# not_required both mean no concern, in_progress is a fixed mid-level
# concern regardless of how close someone is to a deadline.

## TODO is FS even relevant?
FS_STATUS_CONCERN = {
    "achieved": 0.0,
    "not_required": 0.0,
    "in_progress": 0.5,
}


def calculate_risk_score(df, weights=None, caps=None):
    """
    Returns a copy of df with risk_score (0-1) and risk_band
    (Low/Medium/High) columns added, using the weighted formula above.
    """
    weights = weights or WEIGHTS
    caps = caps or CAPS
    df = df.copy()

    attendance_component = (100 - df["attendance_pct"]) / 100
    otj_component = np.clip(df["otj_deficit_hours"].clip(lower=0) / caps["otj_deficit_hours"], 0, 1)
    outstanding_component = np.clip(df["assignments_outstanding_count"] / caps["assignments_outstanding_count"], 0, 1)
    late_component = np.clip(df["assignments_late_count"] / caps["assignments_late_count"], 0, 1)
    onefile_component = np.clip(df["days_since_onefile_update"] / caps["days_since_onefile_update"], 0, 1)
    mentoring_component = np.clip(df["days_since_last_mentoring"] / caps["days_since_last_mentoring"], 0, 1)
    review_component = np.clip(df["days_since_last_progress_review"] / caps["days_since_last_progress_review"], 0, 1)
    fs_maths_component = df["fs_maths_status"].map(FS_STATUS_CONCERN)
    fs_english_component = df["fs_english_status"].map(FS_STATUS_CONCERN)

    risk_score = (
        weights["attendance_pct"] * attendance_component
        + weights["otj_deficit_hours"] * otj_component
        + weights["assignments_outstanding_count"] * outstanding_component
        + weights["assignments_late_count"] * late_component
        + weights["days_since_onefile_update"] * onefile_component
        + weights["days_since_last_mentoring"] * mentoring_component
        + weights["days_since_last_progress_review"] * review_component
        + weights["fs_maths_status"] * fs_maths_component
        + weights["fs_english_status"] * fs_english_component
    )

    df["risk_score"] = np.round(risk_score, 3)
    # Starting thresholds
    df["risk_band"] = pd.cut(
        df["risk_score"],
        bins=[-0.01, 0.3, 0.6, 1.01],
        labels=["Low", "Medium", "High"],
    )
    return df


if __name__ == "__main__":
    df = pd.read_csv("synthetic_apprentice_risk_data.csv")
    scored = calculate_risk_score(df)

    print(f"Scored {len(scored)} apprentices\n")
    print("Risk band counts:")
    print(scored["risk_band"].value_counts())

    print("\nTop 10 highest risk:")
    top10 = scored.sort_values("risk_score", ascending=False).head(10)
    print(
        top10[
            [
                "synthetic_apprentice_id",
                "cohort",
                "attendance_pct",
                "otj_deficit_hours",
                "days_since_last_progress_review",
                "days_since_last_mentoring",
                "assignments_outstanding_count",
                "assignments_late_count",
                "risk_score",
                "risk_band",
            ]
        ].to_string(index=False)
    )

    out_path = "synthetic_apprentice_risk_scored.csv"
    scored.to_csv(out_path, index=False)
    print(f"\nSaved scored data -> {out_path}")
