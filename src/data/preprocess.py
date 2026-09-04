"""Audio preprocessing: load, trim silence, normalize, pad/truncate to a fixed length.

Every clip — regardless of source dataset, native sample rate, or duration — comes out of
`preprocess_audio` as the same fixed-length, unit-scaled waveform, which is the precondition
for building uniform-shape feature tensors in `src/features/extract.py`.
"""
from __future__ import annotations

import numpy as np
import librosa

# Chosen once here and imported everywhere else so every stage of the pipeline agrees.
TARGET_SR = 16_000          # speech content is dominated by energy below ~8 kHz (Nyquist here);
                             # 16 kHz is standard for speech tasks and a third the compute of 48 kHz.
TARGET_DURATION_SEC = 2.5   # NOTE: raw file duration is a misleading basis for this (RAVDESS clips
                             # average 3.7s raw vs TESS's 2.1s purely due to dataset-level leading/
                             # trailing silence padding, not more speech). Measured AFTER silence
                             # trimming, both datasets converge to ~1.8-1.9s of actual content; 2.5s
                             # covers >96% of trimmed clips with no truncation (see notebooks/01).
TARGET_LENGTH = int(TARGET_SR * TARGET_DURATION_SEC)
TRIM_TOP_DB = 25            # anything quieter than (peak - 25dB) is treated as leading/trailing silence


def load_audio(filepath: str, sr: int = TARGET_SR) -> np.ndarray:
    """Load an audio file, resampling to `sr` and collapsing to mono."""
    y, _ = librosa.load(filepath, sr=sr, mono=True)
    return y


def trim_silence(y: np.ndarray, top_db: float = TRIM_TOP_DB) -> np.ndarray:
    """Strip leading/trailing silence. Falls back to the original signal if trimming
    would remove everything (can happen on very quiet or very short clips)."""
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return trimmed if trimmed.size > 0 else y


def normalize(y: np.ndarray) -> np.ndarray:
    """Peak-normalize to [-1, 1]. Leaves silent/zero signals untouched (avoids div-by-zero)."""
    peak = np.max(np.abs(y))
    return y / peak if peak > 1e-8 else y


def pad_or_truncate(y: np.ndarray, target_length: int = TARGET_LENGTH) -> np.ndarray:
    """Center-pad short clips with zeros, or truncate long clips from the center-out window
    starting at 0 (simple left-truncation — speech content in these datasets is front-loaded,
    no leading silence remains after trimming)."""
    if len(y) >= target_length:
        return y[:target_length]
    pad_width = target_length - len(y)
    return np.pad(y, (0, pad_width), mode="constant")


def preprocess_audio(filepath: str, sr: int = TARGET_SR, target_length: int = TARGET_LENGTH) -> np.ndarray:
    """Full pipeline: load -> trim silence -> normalize -> pad/truncate. Always returns an
    array of exactly `target_length` samples."""
    y = load_audio(filepath, sr=sr)
    y = trim_silence(y)
    y = normalize(y)
    y = pad_or_truncate(y, target_length)
    assert len(y) == target_length, f"expected {target_length} samples, got {len(y)}"
    return y
