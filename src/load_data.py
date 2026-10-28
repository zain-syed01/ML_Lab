"""Day 1 — load & clean AI4I 2020, then train/test split.

Hints (implement yourself):
- Read the CSV from data/raw/ with pandas.read_csv
- Inspect dtypes, missing values, class balance of the failure target
- Decide which columns are features vs. IDs / leaky labels
  (UDI, Product ID, and failure-type columns can leak the answer —
   for binary "Machine failure", drop those type columns)
- Encode Product Type if you keep it (OneHotEncoder or pd.get_dummies)
- train_test_split with stratify=y so rare failures stay balanced
- Optionally save a cleaned frame under data/processed/
"""

from pathlib import Path

# TODO: DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "ai4i2020.csv"


def load_raw(path: Path | None = None):
    """Load the raw AI4I CSV. Raise a clear error if the file is missing."""
    raise NotImplementedError("Day 1: implement load_raw")


def clean_and_split(df, test_size: float = 0.2, random_state: int = 42):
    """Return X_train, X_test, y_train, y_test after cleaning + encoding."""
    raise NotImplementedError("Day 1: implement clean_and_split")


if __name__ == "__main__":
    # Smoke-check: load → print shape / failure rate → split → print shapes
    raise NotImplementedError("Day 1: wire load_raw + clean_and_split here")
