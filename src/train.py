"""Day 2 — train supervised model(s), CV, metrics, plots.

Hints (implement yourself):
- Start with LogisticRegression (class_weight='balanced' helps rare failures)
  and/or RandomForestClassifier
- Use Pipeline(preprocessor, model) if you need scaling / one-hot in-fold
- cross_val_score or cross_validate on the *training* set only
- Hold out test set for final accuracy / precision / recall / F1
- Confusion matrix: ConfusionMatrixDisplay.from_predictions
- Feature importance: RF .feature_importances_ or LogReg .coef_
- Save figures under artifacts/ (that folder is gitignored; fine for local demos)
- README: one short paragraph on train vs. CV vs. test (overfitting note)
"""

# TODO: from sklearn.linear_model import LogisticRegression
# TODO: from sklearn.ensemble import RandomForestClassifier
# TODO: from sklearn.model_selection import cross_validate
# TODO: from sklearn.metrics import classification_report, ConfusionMatrixDisplay


def train_and_evaluate():
    """Load data via load_data, fit model(s), print metrics, write plots."""
    raise NotImplementedError("Day 2: implement train_and_evaluate")


if __name__ == "__main__":
    train_and_evaluate()
