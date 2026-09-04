"""Data-prep helpers bridging the framework-agnostic feature arrays (data/processed/features/*.npy
-- built once by src/features/extract.py, reused unchanged here) to what the Keras models expect.
"""
from __future__ import annotations

import numpy as np


def add_channel_dim(mel_array: np.ndarray) -> np.ndarray:
    """(N, n_mels, T) -> (N, n_mels, T, 1). Keras's Conv2D expects an explicit channel axis,
    channels-LAST by default -- our mel-spectrograms are single-channel, so this just appends a
    size-1 axis."""
    return mel_array[..., np.newaxis]


def zero_out_padding(mfcc_array: np.ndarray, valid_lengths: np.ndarray) -> np.ndarray:
    """Explicitly zero every frame beyond each sample's real length, so Keras's
    `layers.Masking(mask_value=0.)` can auto-detect padding by exact-zero rows (see
    src/models.py's module docstring for why this only applies to the LSTM-only path, not
    the hybrid). Returns a COPY -- never mutates the cached .npy array in place."""
    out = mfcc_array.copy()
    for i, length in enumerate(valid_lengths):
        out[i, int(length):, :] = 0.0
    return out


def compute_class_weight_dict(labels: np.ndarray, num_classes: int) -> dict[int, float]:
    """Inverse-frequency class weights (`calm` has ~1/3 the support of every other class), shaped
    into the {class_index: weight} dict Keras's `model.fit(class_weight=...)` expects natively."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = 1.0 / counts
    weights = weights * (num_classes / weights.sum())
    return {i: float(w) for i, w in enumerate(weights)}
