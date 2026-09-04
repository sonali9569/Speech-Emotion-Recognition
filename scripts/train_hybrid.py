"""Stage 7 — CNN-LSTM hybrid + augmentation.
Run with the Keras venv: source .venv_keras/bin/activate && python3 scripts/train_hybrid.py
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.parse import LABEL_TO_IDX
from src.features.augment import build_augmented_train_features
from src.models import build_cnn_lstm_hybrid
from src.data.keras_prep import add_channel_dim
from src.training import train_model, get_predictions, plot_training_curves
from src.evaluation import evaluate, print_summary, save_result, plot_confusion_matrix

tf.random.set_seed(42)
np.random.seed(42)

df = pd.read_csv("data/processed/metadata.csv")
mel = np.load("data/processed/features/mel.npy")
valid_lengths = np.load("data/processed/features/valid_lengths.npy")

train_mask = (df["split"] == "train").to_numpy()
val_mask = (df["split"] == "val").to_numpy()
test_mask = (df["split"] == "test").to_numpy()

train_df = df[train_mask].reset_index(drop=True)
train_labels_orig = train_df["emotion"].map(LABEL_TO_IDX).to_numpy()
print(f"building augmented training features for {len(train_df)} train clips...")
mfcc_aug, mel_aug, labels_aug, valid_lengths_aug = build_augmented_train_features(
    train_df["filepath"].tolist(), train_labels_orig, n_jobs=10
)

mel_train_full = np.concatenate([mel[train_mask], mel_aug], axis=0)
labels_train_full = np.concatenate([train_labels_orig, labels_aug], axis=0)
lengths_train_full = np.concatenate([valid_lengths[train_mask], valid_lengths_aug], axis=0)
print(f"final training set: {len(mel_train_full)} clips ({len(train_df)} original + {len(labels_aug)} augmented)")

y_val = df.loc[val_mask, "emotion"].map(LABEL_TO_IDX).to_numpy()
y_test = df.loc[test_mask, "emotion"].map(LABEL_TO_IDX).to_numpy()

x_train = [add_channel_dim(mel_train_full), lengths_train_full.astype(np.int32)]
x_val = [add_channel_dim(mel[val_mask]), valid_lengths[val_mask].astype(np.int32)]
x_test = [add_channel_dim(mel[test_mask]), valid_lengths[test_mask].astype(np.int32)]

model = build_cnn_lstm_hybrid(n_mels=128, seq_len=79, lstm_hidden=128, num_classes=8, dropout=0.3, use_attention=False)
model, history = train_model(
    model, x_train, labels_train_full, x_val, y_val,
    epochs=60, lr=1e-3, patience=8,
    checkpoint_path="models/cnn_lstm_hybrid.keras",
)

plot_training_curves(history, title="CNN-LSTM hybrid (Keras)", save_path="reports/figures/training_curves_hybrid.png")

y_val_true, y_val_pred = get_predictions(model, x_val, y_val)
y_test_true, y_test_pred = get_predictions(model, x_test, y_test)

val_groups = df.loc[val_mask, "dataset"].to_numpy()
test_groups = df.loc[test_mask, "dataset"].to_numpy()

val_result = evaluate(y_val_true, y_val_pred, groups=val_groups)
test_result = evaluate(y_test_true, y_test_pred, groups=test_groups)

print_summary("Hybrid (Keras) - VAL", val_result)
print()
print_summary("Hybrid (Keras) - TEST", test_result)

save_result({"history": history, "val": val_result, "test": test_result}, "results/hybrid.json")
plot_confusion_matrix(test_result["confusion_matrix"], title="CNN-LSTM hybrid (Keras) - test confusion matrix", save_path="reports/figures/confusion_matrix_hybrid_test.png")
print("\nsaved models/cnn_lstm_hybrid.keras, results/hybrid.json")
