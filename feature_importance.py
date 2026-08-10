# Feature importance for the rules-based scorer
#
# Two different questions, both worth answering:
# 1. "Leave-one-out band impact" - how many apprentices change risk_band
#    if this factor's weight is zeroed out? This shows real-world impact,
#    not just the number you assigned in WEIGHTS.
# 2. "Contribution spread" - how much does each factor's weighted
#    contribution actually vary across apprentices? A factor everyone
#    scores similarly on can't be doing much work distinguishing people,
#    regardless of its weight.

import numpy as np
import pandas as pd
from calculate_risk_score import calculate_risk_score, WEIGHTS, CAPS, FS_STATUS_CONCERN

df = pd.read_csv("synthetic_apprentice_risk_data.csv")
baseline = calculate_risk_score(df)

results = []
for factor in WEIGHTS:
    zeroed_weights = {**WEIGHTS, factor: 0.0}
    variant = calculate_risk_score(df, weights=zeroed_weights)

    band_changed = (variant["risk_band"] != baseline["risk_band"]).sum()

    # Recompute this factor's own weighted contribution per row, to get
    # its spread across the population (independent of leave-one-out)
    components = {
        "attendance_pct": (100 - df["attendance_pct"]) / 100,
        "otj_deficit_hours": np.clip(df["otj_deficit_hours"].clip(lower=0) / CAPS["otj_deficit_hours"], 0, 1),
        "assignments_outstanding_count": np.clip(df["assignments_outstanding_count"] / CAPS["assignments_outstanding_count"], 0, 1),
        "assignments_late_count": np.clip(df["assignments_late_count"] / CAPS["assignments_late_count"], 0, 1),
        "days_since_onefile_update": np.clip(df["days_since_onefile_update"] / CAPS["days_since_onefile_update"], 0, 1),
        "days_since_last_mentoring": np.clip(df["days_since_last_mentoring"] / CAPS["days_since_last_mentoring"], 0, 1),
        "days_since_last_progress_review": np.clip(df["days_since_last_progress_review"] / CAPS["days_since_last_progress_review"], 0, 1),
        "fs_maths_status": df["fs_maths_status"].map(FS_STATUS_CONCERN),
        "fs_english_status": df["fs_english_status"].map(FS_STATUS_CONCERN),
    }
    weighted_contribution = WEIGHTS[factor] * components[factor]

    results.append({
        "factor": factor,
        "weight": WEIGHTS[factor],
        "apprentices_changed_band": band_changed,
        "pct_changed_band": round(band_changed / len(df) * 100, 1),
        "contribution_std": round(weighted_contribution.std(), 4),
    })

importance = pd.DataFrame(results).sort_values("apprentices_changed_band", ascending=False)
print(importance.to_string(index=False))
