"""Feature extraction: MFCC(+delta+delta-delta) sequences for the LSTM path, and log-mel
spectrograms for the CNN path. Both are derived from the same preprocessed, fixed-length waveform
(`src.data.preprocess.preprocess_audio`), so every clip yields identically-shaped tensors.

STFT parameters (n_fft=2048, hop_length=512) are the standard speech-processing choice: at typical
speech sample rates this gives a ~10-30ms analysis frame with 75% overlap — long enough to resolve a
pitch period and formant structure, short enough that phonemes don't blur together across a frame.
"""
from __future__ import annotations

import numpy as np
import librosa
from joblib import Parallel, delayed

from src.data.preprocess import TARGET_SR, preprocess_audio

N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 40
N_MELS = 128


def extract_mfcc_sequence(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """MFCC + delta + delta-delta, stacked along the feature axis and transposed to
    (time_steps, features) — the shape an LSTM expects (batch, seq_len, input_size)."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    stacked = np.concatenate([mfcc, delta, delta2], axis=0)  # (3*N_MFCC, T)
    return stacked.T.astype(np.float32)  # (T, 3*N_MFCC)


def extract_mel_spectrogram(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Log-power mel-spectrogram, shape (n_mels, time_steps) — treated as a single-channel
    image by the CNN path."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def normalize_utterance(x: np.ndarray, valid_length: int, time_axis: int = 0, eps: float = 1e-8) -> np.ndarray:
    """Per-utterance CMVN (Cepstral Mean and Variance Normalization): zero-mean, unit-variance per
    feature channel, computed ONLY from the real (unpadded) frames -- using the padded region would
    pull every clip's stats toward 0, understating real variance and making short clips (more
    padding) normalize differently from long ones for no acoustic reason.

    Why this matters for speaker independence specifically: raw MFCC/mel values carry substantial
    speaker identity (vocal tract shape, natural loudness) mixed in with emotion content. Two
    speakers expressing the same emotion produce different absolute numbers just because their
    voices are physically different -- CMVN removes each clip's own absolute level/scale, leaving
    (mostly) relative, more speaker-invariant shape.

    `time_axis`: which axis of `x` is time (0 for (T, features) like MFCC; -1 for (features, T)
    like a mel-spectrogram). Stats are computed per-channel across the valid time steps.
    """
    valid_length = max(int(valid_length), 1)
    if time_axis == 0:
        valid = x[:valid_length]
        mean = valid.mean(axis=0, keepdims=True)
        std = valid.std(axis=0, keepdims=True)
    else:
        valid = x[..., :valid_length]
        mean = valid.mean(axis=-1, keepdims=True)
        std = valid.std(axis=-1, keepdims=True)
    return ((x - mean) / (std + eps)).astype(np.float32)


def compute_valid_length(y: np.ndarray, hop_length: int = HOP_LENGTH, total_frames: int | None = None) -> int:
    """How many of a clip's feature-sequence time steps are real signal vs. trailing
    zero-padding from `pad_or_truncate`. Used to mask padded frames out of the LSTM's recurrence
    (see src/training.py / src/models/lstm.py, src/models/hybrid.py) instead of letting them
    silently dilute the final hidden state.

    Works directly off the waveform's trailing zeros rather than threading duration metadata
    through every transform (trim/normalize/augment) -- `pad_or_truncate` always pads with exact
    0.0 at the tail, and real audio essentially never hits exact 0.0, so this is robust to
    augmentation (pitch shift, time stretch, noise) as well as the original clips."""
    nonzero = np.flatnonzero(np.abs(y) > 1e-6)
    valid_samples = int(nonzero[-1]) + 1 if nonzero.size > 0 else len(y)
    valid_frames = 1 + valid_samples // hop_length
    if total_frames is not None:
        valid_frames = min(valid_frames, total_frames)
    return max(valid_frames, 1)  # never report zero valid frames -- LSTM needs at least one step


def _extract_both(filepath: str) -> tuple[np.ndarray, np.ndarray, int]:
    y = preprocess_audio(filepath)
    mfcc = extract_mfcc_sequence(y)
    mel = extract_mel_spectrogram(y)
    valid_length = compute_valid_length(y, total_frames=mfcc.shape[0])
    return mfcc, mel, valid_length


def batch_extract(filepaths: list[str], n_jobs: int = -1, verbose: int = 5):
    """Preprocess + extract both feature types (+ valid sequence length) for every filepath, in
    parallel across CPU cores. Returns (mfcc_array, mel_array, valid_lengths): shapes
    (N, T_mfcc, 120), (N, N_MELS, T_mel), (N,)."""
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_extract_both)(fp) for fp in filepaths
    )
    mfcc_arr = np.stack([r[0] for r in results])
    mel_arr = np.stack([r[1] for r in results])
    valid_lengths = np.array([r[2] for r in results], dtype=np.int64)
    return mfcc_arr, mel_arr, valid_lengths
