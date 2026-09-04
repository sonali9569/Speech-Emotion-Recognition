"""Stage 5 — train and evaluate the SVM baseline.
Run: python3 scripts/train_svm_baseline.py
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from src.baseline import get_splits, train_svm
from src.evaluation import evaluate, print_summary, save_result, plot_confusion_matrix

df = pd.read_csv("data/processed/metadata.csv")
mfcc = np.load("data/processed/features/mfcc.npy")

X_train, y_train, X_val, y_val, X_test, y_test = get_splits(mfcc, df)
print("shapes:", X_train.shape, X_val.shape, X_test.shape)

model, scaler, best_params = train_svm(X_train, y_train, X_val, y_val)
print("best hyperparameters (selected on val):", best_params)

X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)
val_pred = model.predict(X_val_s)
test_pred = model.predict(X_test_s)

val_groups = df[df["split"] == "val"]["dataset"].to_numpy()
test_groups = df[df["split"] == "test"]["dataset"].to_numpy()

val_result = evaluate(y_val, val_pred, groups=val_groups)
test_result = evaluate(y_test, test_pred, groups=test_groups)

print_summary("SVM baseline - VAL", val_result)
print()
print_summary("SVM baseline - TEST", test_result)

save_result({"best_params": best_params, "val": val_result, "test": test_result}, "results/svm_baseline.json")
plot_confusion_matrix(test_result["confusion_matrix"], title="SVM baseline - test set confusion matrix", save_path="reports/figures/confusion_matrix_svm_test.png")
print("\nsaved results/svm_baseline.json, reports/figures/confusion_matrix_svm_test.png")
