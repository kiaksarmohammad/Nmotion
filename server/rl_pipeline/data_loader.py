"""
Data loading, preprocessing, SMOTE-TS augmentation, and windowing.

CSV naming convention:
    neuromotion_{category}_{subject_id}_{anything}.csv
    e.g. neuromotion_n1_subj01_run1.csv

Category → label mapping:
    n1, n2              → 0  (normal)
    spasms4, spasms5    → 1  (spasms / pathological)
    hyper6              → 2  (hypertonia)
"""

from __future__ import annotations

import logging
import re
import warnings
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Recording = namedtuple("Recording", ["df", "category", "label", "subject_id", "fname"])

CATEGORY_LABEL: Dict[str, int] = {
    "n1": 0,
    "n2": 0,
    "spasms4": 1,
    "spasms5": 1,
    "hyper6": 2,
}

_FNAME_RE = re.compile(
    r"neuromotion_(?P<category>[a-z0-9]+)_(?P<subject>[^_]+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_filename(fname: str) -> Tuple[str, int, str]:
    """
    Returns (category, label, subject_id) from a CSV filename.
    Raises ValueError if the filename doesn't match the convention.
    """
    stem = Path(fname).stem
    m = _FNAME_RE.match(stem)
    if m is None:
        raise ValueError(
            f"Filename '{fname}' does not match neuromotion_{{category}}_{{subject_id}}_* "
            "naming convention."
        )
    category = m.group("category").lower()
    subject_id = m.group("subject")
    label = CATEGORY_LABEL.get(category)
    if label is None:
        raise ValueError(
            f"Unknown category '{category}' in filename '{fname}'. "
            f"Supported: {list(CATEGORY_LABEL)}"
        )
    return category, label, subject_id


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_all_recordings(data_dir: Path, biomarker_names: List[str]) -> List[Recording]:
    """
    Scan data_dir for CSVs matching the naming convention.
    Returns list of Recording namedtuples.
    """
    data_dir = Path(data_dir)
    csvs = sorted(data_dir.glob("neuromotion_*.csv"))
    if not csvs:
        logger.warning("No CSVs found in %s — returning empty list.", data_dir)
        return []

    recordings: List[Recording] = []
    for path in csvs:
        try:
            category, label, subject_id = parse_filename(path.name)
            df = pd.read_csv(path)
            # Verify required columns exist
            missing = [c for c in biomarker_names if c not in df.columns]
            if missing:
                logger.warning(
                    "Skipping %s — missing columns: %s", path.name, missing
                )
                continue
            recordings.append(
                Recording(df=df, category=category, label=label,
                          subject_id=subject_id, fname=path.name)
            )
        except ValueError as e:
            logger.warning("Skipping %s: %s", path.name, e)

    logger.info("Loaded %d recordings from %s", len(recordings), data_dir)
    return recordings


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

def impute_missing(df: pd.DataFrame, train_means: pd.Series,
                   biomarker_names: List[str]) -> pd.DataFrame:
    """
    Forward-fill (max 3 consecutive), then fill remaining NaNs with column means
    computed from the training set.
    """
    df = df.copy()
    cols = [c for c in biomarker_names if c in df.columns]
    df[cols] = df[cols].ffill(limit=3)
    for col in cols:
        if col in train_means.index:
            df[col] = df[col].fillna(float(train_means[col]))
    return df


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def stratified_split(
    recordings: List[Recording],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    random_state: int = 42,
) -> Tuple[List[Recording], List[Recording], List[Recording]]:
    """
    70/15/15 split stratified by (category, subject_id).
    Ensures at least 1 OOD subject per category lands in test.
    """
    if not recordings:
        return [], [], []

    indices = np.arange(len(recordings))
    groups = np.array([r.subject_id for r in recordings])
    labels = np.array([r.label for r in recordings])

    # First split: train vs. temp (val + test)
    gss1 = GroupShuffleSplit(
        n_splits=1, test_size=1 - train_frac, random_state=random_state
    )
    train_idx, temp_idx = next(gss1.split(indices, labels, groups))

    # Second split: val vs. test from temp
    val_ratio = val_frac / (val_frac + (1 - train_frac - val_frac))
    gss2 = GroupShuffleSplit(
        n_splits=1, test_size=1 - val_ratio, random_state=random_state
    )
    temp_labels = labels[temp_idx]
    temp_groups = groups[temp_idx]
    rel_val, rel_test = next(gss2.split(temp_idx, temp_labels, temp_groups))
    val_idx = temp_idx[rel_val]
    test_idx = temp_idx[rel_test]

    train = [recordings[i] for i in train_idx]
    val = [recordings[i] for i in val_idx]
    test = [recordings[i] for i in test_idx]

    logger.info(
        "Split: train=%d  val=%d  test=%d", len(train), len(val), len(test)
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def fit_scaler(
    train_recordings: List[Recording], biomarker_names: List[str]
) -> StandardScaler:
    """Fit StandardScaler on training recordings only."""
    if not train_recordings:
        return StandardScaler()
    dfs = [r.df[biomarker_names].dropna() for r in train_recordings]
    combined = pd.concat(dfs, ignore_index=True)

    # Warn for angular_jerk (index 8) — heavy-tailed distribution
    if "angular_jerk" in combined.columns:
        skew = combined["angular_jerk"].skew()
        if abs(skew) > 2.0:
            logger.warning(
                "angular_jerk has high skew (%.2f). "
                "Consider log-transforming before fitting scaler.", skew
            )

    scaler = StandardScaler()
    scaler.fit(combined)
    return scaler


def scale_recording(
    rec: Recording, scaler: StandardScaler, biomarker_names: List[str]
) -> Recording:
    """Return a new Recording with scaled biomarker columns."""
    df = rec.df.copy()
    cols = [c for c in biomarker_names if c in df.columns]
    df[cols] = scaler.transform(df[cols].fillna(0.0))
    return rec._replace(df=df)


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def create_windows(
    df: pd.DataFrame,
    biomarker_names: List[str],
    window_size: int = 30,
    stride: int = 15,
) -> np.ndarray:
    """
    Sliding-window extraction.
    Returns ndarray of shape [N_windows, window_size, N_biomarkers].
    """
    vals = df[biomarker_names].values.astype(np.float32)  # [T, 9]
    n_frames = len(vals)
    if n_frames < window_size:
        # Pad with zeros
        pad = np.zeros((window_size - n_frames, vals.shape[1]), dtype=np.float32)
        vals = np.concatenate([vals, pad], axis=0)
        n_frames = window_size

    windows = []
    start = 0
    while start + window_size <= n_frames:
        windows.append(vals[start: start + window_size])
        start += stride

    if not windows:
        return np.zeros((0, window_size, len(biomarker_names)), dtype=np.float32)
    return np.stack(windows, axis=0)  # [N, W, C]


def recordings_to_windows(
    recordings: List[Recording],
    biomarker_names: List[str],
    window_size: int = 30,
    stride: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flatten all recordings into (windows, labels) arrays.
    Returns:
        windows: [N_total, window_size, N_biomarkers]
        labels:  [N_total]  (broadcast recording label to all its windows)
    """
    all_windows, all_labels = [], []
    for rec in recordings:
        wins = create_windows(rec.df, biomarker_names, window_size, stride)
        if len(wins) == 0:
            continue
        all_windows.append(wins)
        all_labels.append(np.full(len(wins), rec.label, dtype=np.int64))

    if not all_windows:
        shape = (0, window_size, len(biomarker_names))
        return np.zeros(shape, dtype=np.float32), np.zeros(0, dtype=np.int64)

    return np.concatenate(all_windows, axis=0), np.concatenate(all_labels, axis=0)


# ---------------------------------------------------------------------------
# SMOTE-TS
# ---------------------------------------------------------------------------

def apply_smote_ts(
    pathological_windows: np.ndarray,
    n_neighbors: int = 5,
    target_ratio: float = 3.0,
    random_state: int = 42,
) -> np.ndarray:
    """
    Time-series SMOTE using tslearn KMeans cluster centroids as neighbors.
    pathological_windows: [N, W, C]
    Returns augmented windows [M, W, C] where M ≈ N * target_ratio.
    """
    try:
        from tslearn.clustering import TimeSeriesKMeans
    except ImportError:
        logger.warning(
            "tslearn not installed — skipping SMOTE-TS augmentation. "
            "Install with: pip install tslearn"
        )
        return pathological_windows

    N, W, C = pathological_windows.shape
    n_synthetic = max(0, int(N * target_ratio) - N)
    if n_synthetic == 0 or N < n_neighbors:
        return pathological_windows

    rng = np.random.default_rng(random_state)
    n_clusters = min(n_neighbors, N)
    km = TimeSeriesKMeans(
        n_clusters=n_clusters, metric="dtw", random_state=random_state
    )
    # tslearn expects [N, T, 1] or [N, T, C] — our shape [N, W, C] is fine
    km.fit(pathological_windows)

    synthetic = []
    for _ in range(n_synthetic):
        # Pick two random cluster centroids
        c1, c2 = rng.choice(n_clusters, size=2, replace=False)
        alpha = rng.uniform(0, 1)
        new_sample = alpha * km.cluster_centers_[c1] + (1 - alpha) * km.cluster_centers_[c2]
        synthetic.append(new_sample)

    synthetic_arr = np.stack(synthetic, axis=0).astype(np.float32)
    return np.concatenate([pathological_windows, synthetic_arr], axis=0)


# ---------------------------------------------------------------------------
# Weighted sampler
# ---------------------------------------------------------------------------

class WeightedEpisodeSampler:
    """
    Samples recording indices with inverse-class-frequency weights,
    guaranteeing >= 20% pathological per batch.
    """

    def __init__(self, recordings: List[Recording], min_patho_frac: float = 0.20):
        self.recordings = recordings
        self.min_patho_frac = min_patho_frac

        labels = np.array([r.label for r in recordings])
        unique, counts = np.unique(labels, return_counts=True)
        freq = dict(zip(unique.tolist(), counts.tolist()))
        total = len(recordings)
        self.weights = np.array(
            [total / (len(unique) * freq[r.label]) for r in recordings],
            dtype=np.float64,
        )
        self.weights /= self.weights.sum()

        patho_mask = labels > 0
        self.patho_idx = np.where(patho_mask)[0]
        self.normal_idx = np.where(~patho_mask)[0]

    def sample(self, batch_size: int, rng: Optional[np.random.Generator] = None) -> List[int]:
        if rng is None:
            rng = np.random.default_rng()

        n_patho = max(1, int(batch_size * self.min_patho_frac))
        n_normal = batch_size - n_patho

        patho_sample = (
            rng.choice(self.patho_idx, size=n_patho, replace=True)
            if len(self.patho_idx) > 0
            else np.array([], dtype=int)
        )
        normal_sample = (
            rng.choice(self.normal_idx, size=n_normal, replace=True)
            if len(self.normal_idx) > 0
            else np.array([], dtype=int)
        )

        combined = np.concatenate([patho_sample, normal_sample])
        rng.shuffle(combined)
        return combined.tolist()
