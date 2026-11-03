import time

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from fraudcore.features import add_features, load_raw, save_spec
from threshold import best_threshold, precision_recall_at, total_cost

LABEL = "isFraud"
NON_FEATURES = ["TransactionID", LABEL, "TransactionDT"]


def time_split(df, train_frac=0.6, val_frac=0.2):
    n = len(df)
    ival = int(n * train_frac)
    itest = int(n * (train_frac + val_frac))
    return df.iloc[:ival].copy(), df.iloc[ival:itest].copy(), df.iloc[itest:].copy()


def categorical_columns(df):
    cat_cols = []
    for i in df.columns:
        if i not in NON_FEATURES and not pd.api.types.is_numeric_dtype(df[i]):
            cat_cols.append(i)
    return cat_cols


def fit_categories(train, cat_cols):
    return {c: sorted(train[c].dropna().unique().tolist()) for c in cat_cols}


def apply_categories(df, categories):
    for c, levels in categories.items():
        known = df[c].where(df[c].isin(levels))
        df[c] = pd.Categorical(known, levels)
    return df


def train_model(X_tr, y_tr, X_va, y_va):
    params = dict(
        objective="binary", metric="average_precision",
        learning_rate=0.05, num_leaves=127, min_child_samples=100,
        feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
        lambda_l2=1.0, verbose=-1, seed=42,
    )
    return lgb.train(params, lgb.Dataset(X_tr, y_tr), num_boost_round=3000,
                     valid_sets=[lgb.Dataset(X_va, y_va)],
                     callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])


def main():
    t0 = time.time()
    df = load_raw("data/raw")
    print(f"loaded {len(df):,} rows in {time.time() - t0:.0f}s, fraud rate {df[LABEL].mean():.2%}")

    train, val, test = time_split(df)
    del df
    for part in (train, val, test):
        add_features(part)

    cat_cols = categorical_columns(train)
    categories = fit_categories(train, cat_cols)
    for part in (train, val, test):
        apply_categories(part, categories)

    FEATURES = [c for c in train.columns if c not in NON_FEATURES]
    print(f"{len(FEATURES)} features, {len(cat_cols)} categorical")
    X_tr, y_tr = train[FEATURES], train[LABEL]
    X_va, y_va = val[FEATURES], val[LABEL]
    X_te, y_te = test[FEATURES], test[LABEL]

    model = train_model(X_tr, y_tr, X_va, y_va)
    print(f"best iteration {model.best_iteration}")

    p_va = model.predict(X_va, num_iteration=model.best_iteration)
    p_te = model.predict(X_te, num_iteration=model.best_iteration)
    for name, y, p in [("val", y_va, p_va), ("test", y_te, p_te)]:
        print(f"{name:5s} PR-AUC {average_precision_score(y, p):.3f} (random = {y.mean():.3f})   "
              f"ROC-AUC {roc_auc_score(y, p):.3f}")

    thr, _ = best_threshold(y_va.values, p_va, val["TransactionAmt"].values)
    prec, rec = precision_recall_at(y_te.values, p_te, thr)
    cost = total_cost(y_te.values, p_te, test["TransactionAmt"].values, thr)
    print(f"threshold {thr} -> test precision {prec:.1%}, recall {rec:.1%}, cost ${cost:,.0f}")

    model.save_model("models/model.txt", num_iteration=model.best_iteration)
    save_spec("models/spec.json", FEATURES, categories, thr)
    print("saved models/model.txt and models/spec.json")


if __name__ == "__main__":
    main()
