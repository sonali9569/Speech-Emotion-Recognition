"""SVM baseline: averaged MFCC(+delta+delta2) per clip -> RBF-kernel SVM.

Collapses the (T, 120) MFCC sequence to a single 120-dim vector per clip (per-clip average),
discarding temporal structure entirely. Serves as the floor every temporal/spectro-temporal
model must beat to justify its added complexity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from src.data.parse import LABEL_TO_IDX


def flatten_by_time_average(mfcc_array: np.ndarray) -> np.ndarray:
    """(N, T, 120) -> (N, 120) by averaging over the time axis."""
    return mfcc_array.mean(axis=1)


def get_splits(mfcc_array: np.ndarray, metadata: pd.DataFrame):
    X = flatten_by_time_average(mfcc_array)
    y = metadata["emotion"].map(LABEL_TO_IDX).to_numpy()
    split = metadata["split"].to_numpy()
    return (
        X[split == "train"], y[split == "train"],
        X[split == "val"], y[split == "val"],
        X[split == "test"], y[split == "test"],
    )


def train_svm(X_train, y_train, X_val, y_val, random_state: int = 42):
    """Standardize (fit on train only), then grid-search C/gamma using a fixed train/val split
    (val is never used for gradient/parameter fitting, only for model selection — test stays untouched)."""
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    # PredefinedSplit: -1 marks train-only rows (never used for validation scoring), 0 marks the
    # held-out val fold -- this makes GridSearchCV select on this exact val set, not a fresh CV fold
    # that could quietly reuse rows already held out elsewhere in the pipeline.
    X_combined = np.vstack([X_train_s, X_val_s])
    y_combined = np.concatenate([y_train, y_val])
    test_fold = np.concatenate([np.full(len(X_train_s), -1), np.zeros(len(X_val_s))])
    ps = PredefinedSplit(test_fold)

    param_grid = {"C": [1, 10, 50], "gamma": ["scale", 0.01, 0.001]}
    grid = GridSearchCV(
        SVC(kernel="rbf", class_weight="balanced", random_state=random_state),
        param_grid, cv=ps, scoring="f1_macro", n_jobs=-1,
        refit=False,  # do NOT let sklearn refit the "best" model on train+val: val must stay held
                      # out of training so evaluating on it afterwards is a genuine, unseen-data number.
    )
    grid.fit(X_combined, y_combined)

    best_model = SVC(kernel="rbf", class_weight="balanced", random_state=random_state, **grid.best_params_)
    best_model.fit(X_train_s, y_train)  # fit on train ONLY -- val and test both stay held out
    return best_model, scaler, grid.best_params_
