"""Stage 7.1/7.2 — waveform augmentation, applied to the TRAINING split only.

Every function here takes an already-preprocessed (trimmed/normalized/fixed-length) waveform and
returns a waveform of the SAME fixed length -- pitch shift preserves length by construction, but
time stretch does not, so it's explicitly re-padded/truncated and re-normalized after the transform.
This means an augmented clip can go straight back through `src/features/extract.py` unchanged.
"""
from __future__ import annotations

import numpy as np
import librosa
from joblib import Parallel, delayed

from src.data.preprocess import TARGET_SR, TARGET_LENGTH, preprocess_audio, normalize, pad_or_truncate
from src.features.extract import extract_mfcc_sequence, extract_mel_spectrogram, compute_valid_length

AUGMENTATION_KINDS = ["pitch_shift", "time_stretch", "noise"]


def pitch_shift(y: np.ndarray, sr: int = TARGET_SR, n_steps: float = 2.0) -> np.ndarray:
    """Shift pitch by n_steps semitones (+/-) without changing duration. Simulates natural
    speaker-to-speaker pitch variation the model shouldn't overfit to."""
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)


def time_stretch(y: np.ndarray, rate: float = 1.15) -> np.ndarray:
    """Speed up (rate>1) or slow down (rate<1) without changing pitch. Changes length, so the
    result is re-fixed to TARGET_LENGTH by the caller."""
    return librosa.effects.time_stretch(y=y, rate=rate)


def add_noise(y: np.ndarray, rng: np.random.Generator, noise_factor: float = 0.01) -> np.ndarray:
    """Additive white Gaussian noise, scaled relative to the signal's own peak amplitude (the
    waveform is already peak-normalized to [-1, 1], so a fixed factor is comparable across clips).

    BUG FIXED HERE: this previously drew from `np.random.randn` (NumPy's global RNG) instead of
    the seeded `rng` passed through `augment_waveform` -- so while the noise *magnitude*
    (`noise_factor`) was reproducible, the actual noise vector never was, silently breaking
    reproducibility of the augmented training set across re-runs. Caught by re-running this exact
    pipeline and finding the "same seed" produced different arrays."""
    noise = rng.standard_normal(len(y)).astype(np.float32)
    return y + noise_factor * noise


def augment_waveform(y: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    """Apply one named augmentation, then re-normalize and re-fix length so the output is a
    drop-in replacement for the original preprocessed waveform."""
    if kind == "pitch_shift":
        n_steps = rng.uniform(-3.0, 3.0)
        y_aug = pitch_shift(y, n_steps=n_steps)
    elif kind == "time_stretch":
        rate = rng.uniform(0.85, 1.20)
        y_aug = time_stretch(y, rate=rate)
    elif kind == "noise":
        noise_factor = rng.uniform(0.005, 0.02)
        y_aug = add_noise(y, rng, noise_factor=noise_factor)
    else:
        raise ValueError(f"unknown augmentation kind: {kind}")

    y_aug = normalize(y_aug)
    y_aug = pad_or_truncate(y_aug, TARGET_LENGTH)
    return y_aug.astype(np.float32)


def _extract_one_augmented(filepath: str, kind: str, seed: int):
    rng = np.random.default_rng(seed)
    y = preprocess_audio(filepath)
    y_aug = augment_waveform(y, kind, rng)
    mfcc = extract_mfcc_sequence(y_aug)
    mel = extract_mel_spectrogram(y_aug)
    # NOT the original clip's valid length -- time_stretch changes duration before the final
    # re-pad/truncate, so an augmented copy's real-signal length can differ from its source
    # clip's. Always recompute from the actual augmented waveform.
    valid_length = compute_valid_length(y_aug, total_frames=mfcc.shape[0])
    return mfcc, mel, valid_length


def build_augmented_train_features(
    train_filepaths: list[str],
    train_labels: np.ndarray,
    kinds: list[str] = AUGMENTATION_KINDS,
    n_jobs: int = -1,
    base_seed: int = 42,
):
    """For every training clip, generate one augmented copy per kind in `kinds` (default: all
    three). Returns (mfcc_aug, mel_aug, labels_aug, valid_lengths_aug) — ONLY the augmented copies,
    so the caller concatenates these onto the original training features to get the full augmented
    training set. Never call this on val/test filepaths."""
    jobs = [
        (fp, kind, base_seed + i * len(kinds) + k)
        for i, fp in enumerate(train_filepaths)
        for k, kind in enumerate(kinds)
    ]
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_extract_one_augmented)(fp, kind, seed) for fp, kind, seed in jobs
    )
    mfcc_aug = np.stack([r[0] for r in results])
    mel_aug = np.stack([r[1] for r in results])
    valid_lengths_aug = np.array([r[2] for r in results], dtype=np.int64)
    labels_aug = np.repeat(train_labels, len(kinds))
    return mfcc_aug, mel_aug, labels_aug, valid_lengths_aug
