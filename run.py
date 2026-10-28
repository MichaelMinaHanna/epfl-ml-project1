"""Pipeline for Project 1 (MICHD prediction from BRFSS survey data).

Steps:
  1. Load the data (cached as .npz after the first, slow CSV load).
  2. Preprocess: one-hot encode categorical columns (at most 10 distinct
     values) and add missing-value indicators for continuous ones; then,
     inside each fold, impute missing values with the training median,
     standardize, and add a bias column.
  3. Compare models with 5-fold cross-validation (F1 score), choosing the
     decision threshold on the out-of-fold predictions.
  4. Retrain the best model on all training data and write a submission.

The switches ONE_HOT, IMPUTE and STANDARDIZE reproduce the ablation study
of the report; the default values give the final submission.
"""

import os
import time

import numpy as np

from helpers import load_csv_data, create_csv_submission
from implementations import (
    logistic_regression,
    reg_logistic_regression,
    ridge_regression,
)

DATA_DIR = "dataset"
SEED = 42
K_FOLDS = 5
MAX_ITERS = 1000
GAMMA = 0.1


# ---------------------------------------------------------------- steps activated 
# Allowed us in the report to compute the ablation table

ONE_HOT = True
IMPUTE = "median"        # "zero" or "median"
STANDARDIZE = True


# ---------------------------------------------------------------- data


def load_data():
    """Load train/test data and column names, using a cache if available."""
    cache = os.path.join(DATA_DIR, "cache.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        x_train, x_test, y_train = d["x_train"], d["x_test"], d["y_train"]
        train_ids, test_ids = d["train_ids"], d["test_ids"]
    else:
        print("Loading CSV files (slow the first time)...")
        x_train, x_test, y_train, train_ids, test_ids = load_csv_data(DATA_DIR)
        np.savez(
            cache, x_train=x_train, x_test=x_test, y_train=y_train,
            train_ids=train_ids, test_ids=test_ids,
        )
    with open(os.path.join(DATA_DIR, "x_train.csv")) as f:
        names = np.array(f.readline().strip().split(",")[1:])  # skip Id
    return x_train, x_test, y_train, test_ids, names


# ------------------------------------------------------- preprocessing
MAX_CATEGORIES = 10


def expand_features(x_train, x_test, names):
    """One-hot encode categorical columns and add missing-value indicators
    for continuous ones. Categories are learned on the training set only;
    a category unseen in training gives all zeros."""
    specs = []
    for j, name in enumerate(names):
        col = x_train[:, j]
        values = np.unique(col[~np.isnan(col)])
        if len(values) <= MAX_CATEGORIES:
            for v in values:
                specs.append(("category", j, v, f"{name}={v:g}"))
        else:
            specs.append(("value", j, None, name))
            if np.isnan(col).any():
                specs.append(("missing", j, None, f"{name}_missing"))

    def build(x):
        out = np.empty((len(x), len(specs)), dtype=np.float32)
        for k, (kind, j, v, _) in enumerate(specs):
            if kind == "category":
                out[:, k] = x[:, j] == v
            elif kind == "missing":
                out[:, k] = np.isnan(x[:, j])
            else:
                out[:, k] = x[:, j]
        return out

    new_names = np.array([s[3] for s in specs])
    return build(x_train), build(x_test), new_names

def fit_scaler(x):
    """Learn imputation medians and standardization stats on training data."""
    medians = np.nanmedian(x, axis=0)
    filled = np.where(np.isnan(x), medians, x)
    mean, std = filled.mean(axis=0), filled.std(axis=0)
    std[std == 0] = 1
    return medians, mean, std


def transform(x, scaler):
    """Impute, (optionally) standardize and add a bias column."""
    medians, mean, std = scaler
    fill = medians if IMPUTE == "median" else 0
    x = np.where(np.isnan(x), fill, x)
    if STANDARDIZE:
        x = (x - mean) / std
    return np.c_[np.ones(len(x), dtype=np.float32), x]


# -------------------------------------------------------------- models

RIDGE_LAMBDAS = np.logspace(-8, 0, 9)     
LOGISTIC_LAMBDAS = [1e-4, 1e-3, 1e-2]

MODELS = {}
for lam in RIDGE_LAMBDAS:
    MODELS[f"ridge (lambda={lam:.0e})"] = (
        lambda y, tx, lam=lam: ridge_regression(y, tx, lam)[0]
    )
MODELS["logistic regression"] = lambda y, tx: logistic_regression(
    y, tx, np.zeros(tx.shape[1], dtype=np.float32), MAX_ITERS, GAMMA
)[0]
for lam in LOGISTIC_LAMBDAS:
    MODELS[f"reg. logistic (lambda={lam:.0e})"] = (
        lambda y, tx, lam=lam: reg_logistic_regression(
            y, tx, lam, np.zeros(tx.shape[1], dtype=np.float32), MAX_ITERS, GAMMA
        )[0]
    )

# The best model, "uncomment it" for faster tests. 
#MODELS = {"logistic regression": lambda y, tx: logistic_regression(
#    y, tx, np.zeros(tx.shape[1], dtype=np.float32), MAX_ITERS, GAMMA
#)[0]}
# ---------------------------------------------------------- evaluation


def f1_score(y_true, y_pred):
    """F1 score for y_true in {0, 1} and boolean predictions."""
    positive = y_true == 1
    tp = np.sum(y_pred & positive)
    fp = np.sum(y_pred & ~positive)
    fn = np.sum(~y_pred & positive)
    return 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0


def best_threshold(y, scores):
    """Threshold on scores that maximizes F1 (tries predicting the top
    1% to 50% of scores as positive)."""
    candidates = np.quantile(scores, np.linspace(0.5, 0.99, 50))
    f1s = [f1_score(y, scores >= t) for t in candidates]
    i = int(np.argmax(f1s))
    return candidates[i], f1s[i]


def cross_validate(train_fn, x, y):
    """K-fold CV: out-of-fold scores, then the best threshold and its F1."""
    rng = np.random.default_rng(SEED)
    folds = np.array_split(rng.permutation(len(y)), K_FOLDS)
    scores = np.empty(len(y))
    for k in range(K_FOLDS):
        va = folds[k]
        tr = np.concatenate([folds[i] for i in range(K_FOLDS) if i != k])
        scaler = fit_scaler(x[tr])
        w = train_fn(y[tr], transform(x[tr], scaler))
        scores[va] = transform(x[va], scaler) @ w
        print(f"  fold {k + 1}/{K_FOLDS} done")
    threshold, f1 = best_threshold(y, scores)
    predicted = scores >= threshold                
    accuracy = np.mean(predicted == (y == 1))     
    return threshold, f1, accuracy


# ---------------------------------------------------------------- main


def main():
    x_train, x_test, y_train, test_ids, names = load_data()
    y = (y_train == 1).astype(np.float32)  # labels {-1, 1} -> {0, 1}
    x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)
    if ONE_HOT:
        x_train, x_test, names = expand_features(x_train, x_test, names)
    print(f"Using {len(names)} features, {len(y)} training samples")
  
    results = {}
    for model_name, train_fn in MODELS.items():
        start = time.time()
        threshold, f1, accuracy = cross_validate(train_fn, x_train, y)
        results[model_name] = (f1, threshold)
        print(f"{model_name:42s} CV F1 = {f1:.4f}  accuracy = {accuracy:.4f}  "f"({time.time() - start:.0f}s)")

    best = max(results, key=lambda name: results[name][0])
    f1, threshold = results[best]
    print(f"Best: {best} (CV F1 = {f1:.4f})")

    # Retrain on all training data and predict the test set.
    scaler = fit_scaler(x_train)
    w = MODELS[best](y, transform(x_train, scaler))
    test_scores = transform(x_test, scaler) @ w
    predictions = np.where(test_scores >= threshold, 1, -1)

    os.makedirs("submissions", exist_ok=True)
    create_csv_submission(test_ids, predictions, "submissions/submission.csv")
    print(f"Saved submissions/submission.csv "
          f"({np.mean(predictions == 1):.1%} predicted positive)")


if __name__ == "__main__":
    main()