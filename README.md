# Apprentice At-Risk Flagging Tool

A weekly-review tool that scores apprentices against several factors and
flags those who may need support, so they can be raised with their
mentor before things escalate.

**Status: proof of concept, synthetic data only.** 

## What it does

Scores each apprentice on OTJ deficit, attendance, assignments
late/outstanding, mentoring/review recency, and OneFile update recency,
then bands them Low/Medium/High. Two hard rules (very poor attendance,
or a full calendar month with no mentoring session) force an automatic
High regardless of the score. The dashboard shows results with a
drill-down explaining each flag.

## Files

- `calculate_risk_score.py` - the scoring formula (weights, caps, rules)
- `app.py` - Streamlit dashboard
- `explore.ipynb` - EDA and weight sense-checking

## Setup

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- Weights/caps are my own starting judgement, not yet validated against
  real outcomes
- `lldd_flag` is never used as a scoring input, only checked manually
  once someone's already flagged

[Demo here](https://synthetic-apprentice-risk-tool-lpcxxkwdwt5abru3red5jk.streamlit.app/)
