"""Shared evaluation used by every model in this project (SVM, LSTM, CNN, hybrid) so results are
comparable on identical terms — same metric set, same label ordering, same plotting style. No stage
is allowed to report a bare accuracy number without also going through here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.data.parse import EMOTIONS


def evaluate(y_true, y_pred, labels: list[str] = EMOTIONS, groups=None) -> dict:
    """Return accuracy, full per-class precision/recall/F1, and the raw confusion matrix.

    `groups` (optional, e.g. the 'dataset' column for these rows) additionally breaks accuracy out
    per group. This matters here specifically: TESS (2 controlled lab speakers) is acoustically far
    easier to separate than RAVDESS, so a blended accuracy can look strong purely by getting TESS
    almost perfectly right while doing much worse on RAVDESS — a fact a single headline number hides.
    """
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, labels=list(range(len(labels))), target_names=labels,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    result = {"accuracy": acc, "classification_report": report, "confusion_matrix": cm.tolist()}
    if groups is not None:
        correct = (np.asarray(y_true) == np.asarray(y_pred))
        result["accuracy_by_group"] = (
            pd.Series(correct).groupby(np.asarray(groups)).mean().to_dict()
        )
    return result


def plot_confusion_matrix(cm, labels: list[str] = EMOTIONS, title: str = "", save_path: str | None = None):
    cm = np.asarray(cm)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_norm, annot=cm, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax, cbar=True,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120)
    return fig


def print_summary(name: str, result: dict):
    print(f"=== {name} ===")
    print(f"accuracy: {result['accuracy']:.4f}")
    if "accuracy_by_group" in result:
        print("accuracy by group:", {k: round(v, 4) for k, v in result["accuracy_by_group"].items()})
    rep = result["classification_report"]
    print(f"macro avg F1: {rep['macro avg']['f1-score']:.4f} | weighted avg F1: {rep['weighted avg']['f1-score']:.4f}")
    print(f"{'class':<12}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for label in EMOTIONS:
        r = rep[label]
        print(f"{label:<12}{r['precision']:>10.3f}{r['recall']:>10.3f}{r['f1-score']:>10.3f}{r['support']:>10.0f}")


def save_result(result: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
