"""Stage 6 — train and evaluate the CNN-only model.
Run with the Keras venv: source .venv_keras/bin/activate && python3 scripts/train_cnn_only.py
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.parse import LABEL_TO_IDX
from src.models import build_cnn_classifier
from src.data.keras_prep import add_channel_dim
from src.training import train_model, get_predictions, plot_training_curves
from src.evaluation import evaluate, print_summary, save_result, plot_confusion_matrix

tf.random.set_seed(42)
np.random.seed(42)

df = pd.read_csv("data/processed/metadata.csv")
mel = add_channel_dim(np.load("data/processed/features/mel.npy"))

train_mask = (df["split"] == "train").to_numpy()
val_mask = (df["split"] == "val").to_numpy()
test_mask = (df["split"] == "test").to_numpy()

y = df["emotion"].map(LABEL_TO_IDX).to_numpy()
x_train, y_train = mel[train_mask], y[train_mask]
x_val, y_val = mel[val_mask], y[val_mask]
x_test, y_test = mel[test_mask], y[test_mask]

model = build_cnn_classifier(n_mels=128, seq_len=79, num_classes=8, dropout=0.4)
model, history = train_model(
    model, x_train, y_train, x_val, y_val,
    epochs=60, lr=1e-3, patience=8,
    checkpoint_path="models/cnn_only.keras",
)

plot_training_curves(history, title="CNN-only (Keras)", save_path="reports/figures/training_curves_cnn.png")

y_val_true, y_val_pred = get_predictions(model, x_val, y_val)
y_test_true, y_test_pred = get_predictions(model, x_test, y_test)

val_groups = df.loc[val_mask, "dataset"].to_numpy()
test_groups = df.loc[test_mask, "dataset"].to_numpy()

val_result = evaluate(y_val_true, y_val_pred, groups=val_groups)
test_result = evaluate(y_test_true, y_test_pred, groups=test_groups)

print_summary("CNN-only (Keras) - VAL", val_result)
print()
print_summary("CNN-only (Keras) - TEST", test_result)

save_result({"history": history, "val": val_result, "test": test_result}, "results/cnn_only.json")
plot_confusion_matrix(test_result["confusion_matrix"], title="CNN-only (Keras) - test confusion matrix", save_path="reports/figures/confusion_matrix_cnn_test.png")
print("\nsaved models/cnn_only.keras, results/cnn_only.json")
