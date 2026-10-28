# Industrial Predictive Maintenance Mini-Lab

Personal supervised ML mini-lab: **binary failure prediction** on the
[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
(UCI Machine Learning Repository).

**Framing:** independent industrial ML practice project for internship applications
(e.g. Aramco Summer 2027). Not affiliated with Aramco, NASA, or any employer.

**Stack:** Python · Pandas · scikit-learn · Matplotlib  
**Not in scope (for now):** Streamlit · Petrobras 3W / Volve / Isolation Forest anomaly work

---

## Goal

Predict `Machine failure` (0/1) from process/sensor-style features, with a clean
train/test split, cross-validation, standard classification metrics, and at least
one Matplotlib figure.

---

## Repo layout

```
data/raw/          # place ai4i2020.csv here (not committed)
data/processed/    # optional cleaned outputs
src/
  load_data.py     # Day 1 — load, clean, split
  train.py         # Day 2 — models, CV, metrics
  plots.py         # Day 2 — confusion matrix / feature importance helpers
artifacts/         # local plots (gitignored)
requirements.txt
README.md
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Dataset (locked)

1. Download **AI4I 2020 Predictive Maintenance Dataset** from UCI (or Kaggle mirror of the same UCI set).
2. Save the CSV as `data/raw/ai4i2020.csv`.
3. Confirm roughly **10,000 rows** and columns such as:  
   `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`,
   `Torque [Nm]`, `Tool wear [min]`, `Machine failure`, plus product type / ID
   and failure-type flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`).

Do **not** commit the CSV (see `.gitignore`).

---

## Your plan (execute yourself)

### Day 1 — Data + split

Checklist:

- [ ] Download CSV → `data/raw/ai4i2020.csv`
- [ ] Implement `src/load_data.py`: load, peek (`head`, `info`, class balance)
- [ ] Choose target: `Machine failure`
- [ ] Drop leaky / ID columns for the binary task (product ID / UDI; failure-type flags if predicting overall failure)
- [ ] Encode categorical `Type` if kept
- [ ] `train_test_split(..., stratify=y, random_state=42)`
- [ ] Run: `python -m src.load_data` and print shapes + failure rate

**Hint:** Failure is rare (~3%). Stratify the split. Note imbalance before picking metrics on Day 2.

### Day 2 — Model + metrics + plots

Checklist:

- [ ] Fit **LogisticRegression** and/or **RandomForestClassifier** (start with one)
- [ ] Cross-validation on the **training** set only
- [ ] Final test metrics: accuracy, precision, recall, F1 (+ `classification_report`)
- [ ] Matplotlib: confusion matrix and/or feature importance → `artifacts/`
- [ ] Fill in **Overfitting note** and **How to run** below
- [ ] Draft 3–4 Google X-Y-Z resume bullets (ask for a review when ready)

**Hint:** Accuracy alone is misleading with ~3% positives — lead with precision/recall/F1 (or macro/weighted averages). Consider `class_weight='balanced'`.

### Stretch (only if early)

- [ ] Second model + short comparison table in README
- [ ] Optional FastAPI `POST /predict` (no Streamlit)
- [ ] Simple threshold discussion if you use `predict_proba`

---

## How to run

*(Fill this in after Day 2.)*

```bash
# Expected once implemented:
python -m src.load_data
python -m src.train
```

---

## Overfitting note

*(1–3 sentences after Day 2: compare train vs. CV vs. test; what that implies.)*

---

## Resume bullets (draft later)

Use Google X-Y-Z when Day 2 is done. Ask for a review pass — do not invent metrics.

- _TBD_
- _TBD_
- _TBD_

---

## Status

Scaffold only. Implementation is yours unless you explicitly ask for help on a step
or ask to finish under time pressure.
