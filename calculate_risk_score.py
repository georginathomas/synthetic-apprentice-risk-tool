"""
Apprentice risk scorer

Applies your own weighted scoring formula to the raw synthetic feature
CSV. This is intentionally separate from the data generator - the
generator's own internal risk_score exists only to validate the fake
data, and is never exported. This file is the real scorer you'll
actually use and iterate on.

FS status (English/Maths) has been removed from the scoring formula
entirely, feature importance showed it barely moved anyone (1-3
apprentices out of 266), so it wasn't earning its place. The remaining
seven factors are weighted equally at 1/7 each, rather than ranked
against each other, since the earlier ranking is the part you're least
confident about right now.

On top of the weighted score, HARD_RULES below force an automatic High
regardless of what the weighted score says. These exist for situations
where averaging across factors would mask something that shouldn't be
averaged away, e.g. very poor attendance offset by everything else
looking fine. Each apprentice forced High this way gets a visible
reason in hard_rule_triggered, so it's always clear whether someone's
High because of their score or because of a rule.
"""

import numpy as np
import pandas as pd

FACTORS = [
    "otj_deficit_hours",
    "assignments_outstanding_count",
    "assignments_late_count",
    "attendance_pct",
    "days_since_onefile_update",
    "days_since_last_mentoring",
    "days_since_last_progress_review",
]
WEIGHTS = {factor: 1 / len(FACTORS) for factor in FACTORS}

# Each rule: a label (used in the override reason) and a condition
# function that takes the dataframe and returns a boolean mask. Add or
# remove rules here - no other code needs to change.
def missed_calendar_month(last_date_col, reference_date=None):
    """
    True where the apprentice has gone an entire calendar month with no
    mentoring session - i.e. their last session was before the start of
    the previous calendar month. Being seen at any point last month or
    this month counts as compliant, even if it was weeks ago; this only
    catches a genuinely skipped month, not "hasn't had this month's
    session yet" (which would wrongly flag most people mid-month).
    """
    reference_date = pd.Timestamp(reference_date) if reference_date else pd.Timestamp.today().normalize()
    last_dates = pd.to_datetime(last_date_col)
    start_of_this_month = reference_date.replace(day=1)
    start_of_last_month = start_of_this_month - pd.DateOffset(months=1)
    return last_dates < start_of_last_month


HARD_RULES = [
    {"label": "Attendance below 70%", "condition": lambda df: df["attendance_pct"] < 70},
    {"label": "Went a full calendar month with no mentoring session", "condition": lambda df: missed_calendar_month(df["last_mentoring_date"])},
]

# Caps used to scale each raw value to a 0-1 "concern" range before
# weighting. These are starting assumptions, not fixed rules - adjust
# based on what actually counts as concerning in your programme.
CAPS = {
    "otj_deficit_hours": 50,       # 50+ hours deficit = max concern
    "assignments_outstanding_count": 5,
    "assignments_late_count": 5,
    "days_since_onefile_update": 30,      # a month with no timesheet update = max concern
    "days_since_last_mentoring": 60,
    "days_since_last_progress_review": 84, # 12 weeks in days
}

# FS status -> concern value. No longer used in calculate_risk_score()
# below - kept here only because explore.ipynb still uses it to build
# fs_english_numeric/fs_maths_numeric columns for EDA/plotting.
FS_STATUS_CONCERN = {
    "achieved": 0.0,
    "not_required": 0.0,
    "in_progress": 0.5,
}


def apply_hard_rules(df, rules=None):
    """
    Returns a copy of df with hard_rule_triggered added (a semicolon-
    separated list of reasons, or None) and risk_band overridden to
    High wherever any rule fires - regardless of what the weighted
    score alone would have produced.
    """
    rules = rules or HARD_RULES
    df = df.copy()

    reasons = pd.Series([[] for _ in range(len(df))], index=df.index)
    for rule in rules:
        mask = rule["condition"](df)
        for idx in df.index[mask]:
            reasons.loc[idx].append(rule["label"])

    df["hard_rule_triggered"] = reasons.apply(lambda r: "; ".join(r) if r else None)
    df.loc[df["hard_rule_triggered"].notna(), "risk_band"] = "High"
    return df


def calculate_risk_score(df, weights=None, caps=None, rules=None):
    """
    Returns a copy of df with risk_score (0-1), risk_band
    (Low/Medium/High), and hard_rule_triggered added. risk_band reflects
    both the weighted score and any hard rule overrides - see
    apply_hard_rules() for the override logic.
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

    risk_score = (
        weights["attendance_pct"] * attendance_component
        + weights["otj_deficit_hours"] * otj_component
        + weights["assignments_outstanding_count"] * outstanding_component
        + weights["assignments_late_count"] * late_component
        + weights["days_since_onefile_update"] * onefile_component
        + weights["days_since_last_mentoring"] * mentoring_component
        + weights["days_since_last_progress_review"] * review_component
    )

    df["risk_score"] = np.round(risk_score, 3)
    # Starting thresholds - same as before, worth re-checking against
    # your own sense of what counts as Medium vs High once you've run
    # this against apprentices you already know the story of.
    df["risk_band"] = pd.cut(
        df["risk_score"],
        bins=[-0.01, 0.3, 0.6, 1.01],
        labels=["Low", "Medium", "High"],
    )
    df["risk_band"] = df["risk_band"].astype(object)  # allow override below to assign "High" freely
    df = apply_hard_rules(df, rules=rules)
    return df


if __name__ == "__main__":
    df = pd.read_csv("synthetic_apprentice_risk_data.csv")
    scored = calculate_risk_score(df)

    print(f"Scored {len(scored)} apprentices\n")
    print("Risk band counts:")
    print(scored["risk_band"].value_counts())

    rule_triggered = scored["hard_rule_triggered"].notna()
    print(f"\n{rule_triggered.sum()} apprentices flagged High by a hard rule (not just the weighted score):")
    print(
        scored[rule_triggered][
            ["synthetic_apprentice_id", "cohort", "risk_score", "hard_rule_triggered"]
        ].to_string(index=False)
    )

    print("\nTop 10 highest risk:")
    top10 = scored.sort_values("risk_score", ascending=False).head(10)
    print(
        top10[
            [
                "synthetic_apprentice_id",
                "cohort",
                "attendance_pct",
                "otj_deficit_hours",
                "assignments_outstanding_count",
                "assignments_late_count",
                "risk_score",
                "risk_band",
                "hard_rule_triggered",
            ]
        ].to_string(index=False)
    )

    out_path = "synthetic_apprentice_risk_scored.csv"
    scored.to_csv(out_path, index=False)
    print(f"\nSaved scored data -> {out_path}")
