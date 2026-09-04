"""Inference pipeline: raw audio file -> predicted emotion.

Wraps the exact same preprocessing and feature extraction used at training time
(`src/data/preprocess.py`, `src/features/extract.py`) so a file passed in here goes through
identically to how every training/test clip did -- no separate "demo-only" pipeline to drift out
of sync with what the model was actually trained on.
"""
from __future__ import annotations

import numpy as np
import keras

import src.models  # noqa: F401 -- import for its side effect: registers SequenceMaskLayer and
                    # MaskedAttentionPooling as deserializable, which keras.models.load_model needs
                    # even though this module never calls the builder functions directly.
from src.data.preprocess import preprocess_audio
from src.features.extract import extract_mel_spectrogram, compute_valid_length
from src.data.parse import IDX_TO_LABEL


def load_model(checkpoint_path: str = "models/cnn_lstm_hybrid.keras") -> keras.Model:
    return keras.models.load_model(checkpoint_path)


def predict_emotion(filepath: str, model: keras.Model, confidence_threshold: float = 0.5) -> dict:
    """Run one audio file through the full pipeline and return the predicted label plus the
    full probability distribution over all 8 emotions (so a caller can see how confident/close
    a call was, not just the top-1 label).

    If the top probability is below `confidence_threshold`, `label` is set to "uncertain" instead
    of forcing a low-confidence guess -- `predicted_emotion` still carries the raw argmax so a
    caller can see what the model's best guess *was*, just not treat it as a confident answer.
    Pass confidence_threshold=0.0 to disable (label always equals predicted_emotion)."""
    y = preprocess_audio(filepath)
    mel = extract_mel_spectrogram(y)  # (n_mels, T)
    valid_length = compute_valid_length(y, total_frames=mel.shape[1])

    mel_input = mel[np.newaxis, ..., np.newaxis]  # (1, n_mels, T, 1) -- channels-last, batch of 1
    length_input = np.array([valid_length], dtype=np.int32)

    probs = model.predict([mel_input, length_input], verbose=0)[0]
    pred_idx = int(probs.argmax())
    confidence = float(probs[pred_idx])

    return {
        "predicted_emotion": IDX_TO_LABEL[pred_idx],
        "confidence": confidence,
        "label": IDX_TO_LABEL[pred_idx] if confidence >= confidence_threshold else "uncertain",
        "probabilities": {IDX_TO_LABEL[i]: float(p) for i, p in enumerate(probs)},
    }
