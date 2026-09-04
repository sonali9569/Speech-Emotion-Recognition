"""Parse RAVDESS and TESS folder/filename conventions into a single labeled DataFrame.

RAVDESS filename convention (8 dash-separated numeric identifiers):
    modality-vocalChannel-emotion-intensity-statement-repetition-actor.wav
    e.g. 03-01-06-02-02-01-12.wav -> emotion code 06 = fearful, actor 12 (even = female)

TESS folder convention:
    <SPEAKER>_<emotion>/  where SPEAKER is OAF (older actress) or YAF (younger actress).
    Emotion spelling/casing is inconsistent between the two speakers' folders
    (e.g. "Fear" vs "fear", "Pleasant_surprise" vs "pleasant_surprised") and is
    normalized here to the canonical RAVDESS emotion vocabulary.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import soundfile as sf

# Canonical 8-class emotion vocabulary shared by both datasets.
RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# TESS has no "calm" class distinct from "neutral"; folder names are normalized
# (lowercased, underscores collapsed) before matching against this table.
TESS_EMOTION_MAP = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fearful",
    "happy": "happy",
    "neutral": "neutral",
    "pleasant_surprise": "surprised",
    "pleasant_surprised": "surprised",
    "sad": "sad",
}

EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

# Label<->index mapping used by every model.
LABEL_TO_IDX = {label: i for i, label in enumerate(EMOTIONS)}
IDX_TO_LABEL = {i: label for label, i in LABEL_TO_IDX.items()}


def parse_ravdess_filename(path: Path) -> dict:
    """Extract metadata from one RAVDESS filename. Raises ValueError if malformed."""
    parts = path.stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"Unexpected RAVDESS filename format: {path.name}")
    modality, vocal_channel, emotion_code, intensity, statement, repetition, actor = parts
    if emotion_code not in RAVDESS_EMOTION_MAP:
        raise ValueError(f"Unknown RAVDESS emotion code {emotion_code!r} in {path.name}")
    actor_id = int(actor)
    return {
        "filepath": str(path),
        "dataset": "ravdess",
        "emotion": RAVDESS_EMOTION_MAP[emotion_code],
        "intensity": "normal" if intensity == "01" else "strong",
        "actor": f"ravdess_{actor_id:02d}",
        "gender": "male" if actor_id % 2 == 1 else "female",
    }


def parse_tess_folder(path: Path) -> dict:
    """Extract metadata from one TESS clip using its parent folder name."""
    folder = path.parent.name  # e.g. "OAF_Pleasant_surprise"
    speaker, _, raw_emotion = folder.partition("_")
    normalized = re.sub(r"[^a-z]+", "_", raw_emotion.lower()).strip("_")
    if normalized not in TESS_EMOTION_MAP:
        raise ValueError(f"Unrecognized TESS emotion folder {folder!r} for {path.name}")
    return {
        "filepath": str(path),
        "dataset": "tess",
        "emotion": TESS_EMOTION_MAP[normalized],
        "intensity": "normal",
        "actor": f"tess_{speaker.upper()}",
        "gender": "female",  # both TESS speakers (OAF, YAF) are female
    }


def build_dataset_dataframe(ravdess_root: str | Path, tess_root: str | Path) -> pd.DataFrame:
    """Walk both dataset roots and return one combined (filepath, emotion, ...) DataFrame."""
    ravdess_root, tess_root = Path(ravdess_root), Path(tess_root)

    records = []
    for wav_path in sorted(ravdess_root.rglob("*.wav")):
        records.append(parse_ravdess_filename(wav_path))
    for wav_path in sorted(tess_root.rglob("*.wav")):
        records.append(parse_tess_folder(wav_path))

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise RuntimeError(f"No .wav files found under {ravdess_root} or {tess_root}")

    df["emotion"] = pd.Categorical(df["emotion"], categories=EMOTIONS, ordered=False)
    return df.reset_index(drop=True)


def add_audio_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add duration_sec and sample_rate columns by reading each file's header only
    (soundfile.info does not decode the waveform, so this is fast even for 4k+ files)."""
    durations, sample_rates = [], []
    for filepath in df["filepath"]:
        info = sf.info(filepath)
        durations.append(info.frames / info.samplerate)
        sample_rates.append(info.samplerate)
    df = df.copy()
    df["duration_sec"] = durations
    df["sample_rate"] = sample_rates
    return df
