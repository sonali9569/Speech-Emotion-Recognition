"""Stratified train/val/test split, by emotion label, at a fixed random seed.

Caveat (documented, not hidden): this stratifies by class label only, not by speaker. RAVDESS has
24 actors so a speaker-disjoint split would be possible there, but TESS has only 2 speakers (OAF,
YAF) — holding one out as "test" would remove an entire recording identity's timbre from training
and roughly halve TESS's usable training data, which is a worse trade-off than the label leakage
it would prevent. Evaluation results should be read as "this model discriminates emotion from
acoustics across these speakers," not "generalizes to an unseen speaker."
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def stratified_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> pd.Series:
    """Return a Series aligned to df.index with values in {'train','val','test'}."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9

    train_idx, rest_idx = train_test_split(
        df.index,
        train_size=train_frac,
        stratify=df["emotion"],
        random_state=random_state,
    )
    rest = df.loc[rest_idx]
    val_share_of_rest = val_frac / (val_frac + test_frac)
    val_idx, test_idx = train_test_split(
        rest.index,
        train_size=val_share_of_rest,
        stratify=rest["emotion"],
        random_state=random_state,
    )

    split = pd.Series(index=df.index, dtype=object)
    split.loc[train_idx] = "train"
    split.loc[val_idx] = "val"
    split.loc[test_idx] = "test"
    return split
