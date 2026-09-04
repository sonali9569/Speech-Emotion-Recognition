"""Shared training loop for every neural model in this project (LSTM-only, CNN-only, hybrid), built
on Keras's own `model.fit()` + callbacks: Adam (lr=1e-3, weight_decay=1e-4), grad-norm clipping at
5.0, ReduceLROnPlateau (factor=0.5, patience=3), and early stopping (patience=8, restoring the
best-val-loss weights) -- so training/early-stopping/checkpointing behavior is identical across the
ablation and differences in final metrics come from the architecture, not the training setup.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras


def compile_model(model: keras.Model, lr: float = 1e-3, weight_decay: float = 1e-4, grad_clip: float = 5.0):
    optimizer = keras.optimizers.Adam(learning_rate=lr, weight_decay=weight_decay, clipnorm=grad_clip)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(
    model: keras.Model,
    x_train, y_train,
    x_val, y_val,
    epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 8,
    checkpoint_path: str = "models/checkpoint.keras",
    class_weight: dict | None = None,
    batch_size: int = 32,
):
    """Trains with early stopping (patience epochs with no val_loss improvement, restores the
    best-val-loss weights afterward) and ReduceLROnPlateau. Returns (model, history_dict) with
    normalized key names (train_loss/val_loss/train_acc/val_acc) for plot_training_curves."""
    compile_model(model, lr=lr, weight_decay=weight_decay)
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, min_delta=1e-4),
        keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]

    fit_kwargs = dict(
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    if class_weight is not None:
        fit_kwargs["class_weight"] = class_weight

    keras_history = model.fit(x_train, y_train, **fit_kwargs)

    history = {
        "train_loss": keras_history.history["loss"],
        "val_loss": keras_history.history["val_loss"],
        "train_acc": keras_history.history["accuracy"],
        "val_acc": keras_history.history["val_accuracy"],
    }
    print(f"early stopping / training ended after {len(history['train_loss'])} epochs "
          f"(best val_loss={min(history['val_loss']):.4f})")
    return model, history


def get_predictions(model: keras.Model, x, y_true) -> tuple[np.ndarray, np.ndarray]:
    """Runs inference over `x`, returns (y_true, y_pred) as numpy arrays -- y_true is passed
    through as-is since there's no DataLoader-style iterator to pull labels from."""
    probs = model.predict(x, verbose=0)
    y_pred = probs.argmax(axis=1)
    return np.asarray(y_true), y_pred


def plot_training_curves(history: dict, title: str = "", save_path: str | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title(f"{title} — loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title(f"{title} — accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120)
    return fig
