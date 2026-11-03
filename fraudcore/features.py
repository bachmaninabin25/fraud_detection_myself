"""Feature logic shared by training (notebook) and serving (API / Kafka consumer).

Keeping this in one place means the model sees exactly the same inputs in
production as it did in training.
"""
import json

import numpy as np
import pandas as pd

LABEL = "isFraud"
# Not features: an ID, the label, and raw time (the model would learn the
# calendar trend, which doesn't carry over to future data).
NON_FEATURES = ["TransactionID", LABEL, "TransactionDT"]


def load_raw(data_dir):
    """Load train transactions + identity, left-joined and sorted by time."""
    tx = pd.read_csv(f"{data_dir}/train_transaction.csv")
    idt = pd.read_csv(f"{data_dir}/train_identity.csv")
    df = tx.merge(idt, on="TransactionID", how="left")
    floats = df.select_dtypes("float64").columns
    df[floats] = df[floats].astype("float32")  # halves memory
    return df.sort_values("TransactionDT").reset_index(drop=True)


def add_features(df):
    """Hand-made features on top of the raw columns. Works on 1 row or 1M."""
    dt = df["TransactionDT"]
    amt = df["TransactionAmt"]
    df["amt_log"] = np.log1p(amt).astype("float32")
    df["amt_cents"] = (amt - np.floor(amt)).astype("float32")
    df["hour"] = ((dt // 3600) % 24).astype("float32")
    df["dow"] = ((dt // 86400) % 7).astype("float32")
    return df


def categorical_columns(df):
    return [c for c in df.columns
            if c not in NON_FEATURES and not pd.api.types.is_numeric_dtype(df[c])]


def fit_categories(df, cat_cols):
    """Learn the category levels from the training split only."""
    return {c: sorted(df[c].dropna().unique().tolist()) for c in cat_cols}


def apply_categories(df, categories):
    """Fixed levels for every split; unseen values become NaN."""
    for c, levels in categories.items():
        known = df[c].where(df[c].isin(levels))
        df[c] = pd.Categorical(known, categories=levels)
    return df


def save_spec(path, features, categories, threshold, extra=None):
    spec = {"features": features, "categories": categories,
            "threshold": threshold, **(extra or {})}
    with open(path, "w") as f:
        json.dump(spec, f, indent=1)


def load_spec(path):
    with open(path) as f:
        return json.load(f)


def prepare(df, spec):
    """Raw rows -> model-ready frame (used at serving time)."""
    df = add_features(df.copy())
    for c in spec["features"]:
        if c not in df:
            df[c] = np.nan
    df = apply_categories(df, spec["categories"])
    return df[spec["features"]]
