import numpy as np
import pandas as pd

EPS = 1e-4
N_BINS = 10


def make_edges(train_values, n_bins=N_BINS):
    v = np.asarray(train_values, dtype=float)
    v = v[~np.isnan(v)]
    return np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1])).tolist()


def proportions(values, edges):
    v = np.asarray(values, dtype=float)
    nan = np.isnan(v)
    counts = np.bincount(np.searchsorted(edges, v[~nan], side="right"), minlength=len(edges) + 1).astype(float)
    counts = np.append(counts, nan.sum())
    return counts / max(len(v), 1)


def psi(train_values, live_values):
    edges = make_edges(train_values)
    e = np.clip(proportions(train_values, edges), EPS, None)
    a = np.clip(proportions(live_values, edges), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


if __name__ == "__main__":
    cols = pd.read_csv("data/raw/train_transaction.csv", usecols=["TransactionDT", "TransactionAmt"])
    amt = cols.sort_values("TransactionDT")["TransactionAmt"].to_numpy()
    train_amt, test_amt = amt[: int(len(amt) * 0.6)], amt[int(len(amt) * 0.8):]
    print(f"PSI train vs test amounts:      {psi(train_amt, test_amt):.4f}")
    print(f"PSI train vs test amounts x3:   {psi(train_amt, test_amt * 3):.4f}")
