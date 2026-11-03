"""Fast scoring path for live traffic.

`features.prepare` builds a pandas DataFrame, which costs several milliseconds
per request. Here we fill a numpy array directly, using the same category codes
and engineered features, so it gives identical scores at a fraction of the cost.
"""
import hashlib
import math

import lightgbm as lgb
import numpy as np

from . import features as F


class Scorer:
    def __init__(self, model_path, spec_path):
        self.spec = F.load_spec(spec_path)
        self.booster = lgb.Booster(model_file=model_path)
        self.features = self.spec["features"]
        self.threshold = self.spec["threshold"]
        self.col = {name: j for j, name in enumerate(self.features)}
        self.cat_codes = {c: {v: i for i, v in enumerate(levels)}
                          for c, levels in self.spec["categories"].items()}
        with open(model_path, "rb") as f:
            self.version = hashlib.sha256(f.read()).hexdigest()[:8]

    def vectorize(self, txns):
        """List of raw transaction dicts -> (n, n_features) float array."""
        X = np.full((len(txns), len(self.features)), np.nan)
        col, cat_codes = self.col, self.cat_codes
        for r, t in enumerate(txns):
            row = X[r]
            for k, v in t.items():
                j = col.get(k)
                if j is None or v is None:
                    continue
                codes = cat_codes.get(k)
                if codes is None:
                    row[j] = v
                else:
                    code = codes.get(v)          # unseen category -> stays NaN
                    if code is not None:
                        row[j] = code
            # engineered features, computed in float32 exactly like training
            amt = np.float32(t["TransactionAmt"])
            dt = int(t["TransactionDT"])
            row[col["amt_log"]] = np.log1p(amt)
            row[col["amt_cents"]] = amt - np.floor(amt)
            row[col["hour"]] = (dt // 3600) % 24
            row[col["dow"]] = (dt // 86400) % 7
        return X

    def predict(self, txns):
        return self.booster.predict(self.vectorize(txns), num_threads=1)


def clean_record(rec):
    """Drop missing values so payloads only carry what's known."""
    return {k: v for k, v in rec.items()
            if v is not None and not (isinstance(v, float) and math.isnan(v))}
