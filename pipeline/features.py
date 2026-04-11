"""
Nonlinear dynamics feature extraction from dense optical flow fields.

Converts flow fields [N-1, H, W, 2] into 1D time series, then computes
features that discriminate normal vs. spasms vs. hypertonia movement.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import antropy
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.signal import welch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Flow → time series
# ---------------------------------------------------------------------------

def flow_to_timeseries(flow: np.ndarray) -> Dict[str, np.ndarray]:
    """Convert per-frame flow fields to 1D time series.

    Args:
        flow: [N, H, W, 2] dense optical flow.

    Returns:
        Dict of 1D arrays (length N), keyed by series name.
    """
    # Per-pixel magnitude: sqrt(u^2 + v^2)
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)  # [N, H, W]

    return {
        "magnitude_mean": magnitude.mean(axis=(1, 2)),
        "magnitude_max": magnitude.max(axis=(1, 2)),
        "magnitude_std": magnitude.std(axis=(1, 2)),
    }


# ---------------------------------------------------------------------------
# Individual feature extractors
# ---------------------------------------------------------------------------

def compute_sample_entropy(ts: np.ndarray, order: int = 2) -> float:
    """Sample entropy — regularity/predictability of motion.

    Low = quasi-periodic (seizures). High = complex/irregular (normal).
    """
    return float(antropy.sample_entropy(ts, order=order))


def compute_multiscale_entropy(
    ts: np.ndarray, scales: range = range(1, 21), order: int = 2
) -> np.ndarray:
    """Multiscale entropy — sample entropy at multiple coarse-graining scales.

    Seizures: low at seizure frequency scale, high elsewhere.
    Hypotonic: uniformly low across all scales.

    Returns:
        Array of entropy values, one per scale.
    """
    mse = []
    for scale in scales:
        if scale == 1:
            coarsened = ts
        else:
            # Non-overlapping mean of consecutive blocks
            n = len(ts) - (len(ts) % scale)
            coarsened = ts[:n].reshape(-1, scale).mean(axis=1)

        if len(coarsened) < 20:
            mse.append(np.nan)
            continue

        try:
            mse.append(float(antropy.sample_entropy(coarsened, order=order)))
        except Exception:
            mse.append(np.nan)

    return np.array(mse, dtype=np.float64)


def compute_spectral_entropy(ts: np.ndarray, fps: float) -> float:
    """Spectral entropy — entropy of normalized power spectral density.

    Low = power concentrated in narrow bands (seizures, 1-3 Hz clonic).
    High = diffuse power distribution (normal).
    """
    return float(antropy.spectral_entropy(ts, sf=fps, method="welch", normalize=True))


def compute_dfa(ts: np.ndarray) -> float:
    """Detrended fluctuation analysis — long-range temporal correlations.

    alpha < 0.5: anti-persistent (normal movement).
    alpha > 0.5: persistent (seizures).
    alpha ~ 0.5: white noise (hypotonic).
    """
    return float(antropy.detrended_fluctuation(ts))


def compute_symmetry_index(flow: np.ndarray) -> float:
    """Approximate symmetry index from flow field.

    Splits at vertical midline, compares L2 norms of left vs right.
    Returns value in [0, 1] where 1 = perfectly symmetric.

    Focal seizures break symmetry. Generalized seizures ≈ symmetric.
    """
    _, _, w, _ = flow.shape
    mid = w // 2

    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    left_energy = np.linalg.norm(magnitude[:, :, :mid].ravel())
    right_energy = np.linalg.norm(magnitude[:, :, mid:].ravel())

    total = left_energy + right_energy + 1e-8
    return float(1.0 - abs(left_energy - right_energy) / total)


def compute_peak_frequency(ts: np.ndarray, fps: float) -> float:
    """Dominant frequency in motion power spectrum.

    Clonic seizures: 1-3 Hz. Tonic-clonic: 0.5-1 Hz. Normal: no dominant peak.
    """
    freqs, psd = welch(ts, fs=fps, nperseg=min(256, len(ts)))
    return float(freqs[np.argmax(psd)])


def compute_kinetic_energy(flow: np.ndarray) -> np.ndarray:
    """Mean squared flow magnitude per frame — proxy for kinetic energy.

    Returns 1D array (length N) for use as time series.
    """
    return (flow[..., 0] ** 2 + flow[..., 1] ** 2).mean(axis=(1, 2))


def compute_flow_magnitude_stats(flow: np.ndarray) -> Dict[str, float]:
    """Distribution statistics of per-pixel flow magnitude across all frames."""
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).ravel()
    return {
        "flow_mean": float(magnitude.mean()),
        "flow_std": float(magnitude.std()),
        "flow_skew": float(sp_stats.skew(magnitude)),
        "flow_kurtosis": float(sp_stats.kurtosis(magnitude)),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def extract_features_single(
    flow: np.ndarray, fps: float, video_name: str, group: str
) -> Dict[str, float | str]:
    """Extract all features from a single video's flow fields.

    Args:
        flow: [N, H, W, 2] flow fields.
        fps: Video frame rate.
        video_name: Identifier for this video.
        group: Movement group label.

    Returns:
        Dict of feature values (one row of the final DataFrame).
    """
    ts = flow_to_timeseries(flow)
    ts_mean = ts["magnitude_mean"]
    ke = compute_kinetic_energy(flow)

    # Scalar features
    features: Dict[str, float | str] = {
        "video": video_name,
        "group": group,
        "n_frames": float(len(flow)),
        "fps": fps,
        "sample_entropy": compute_sample_entropy(ts_mean),
        "spectral_entropy": compute_spectral_entropy(ts_mean, fps),
        "dfa_alpha": compute_dfa(ts_mean),
        "symmetry_index": compute_symmetry_index(flow),
        "peak_frequency": compute_peak_frequency(ts_mean, fps),
        "kinetic_energy_mean": float(ke.mean()),
        "kinetic_energy_std": float(ke.std()),
    }

    # Flow magnitude distribution stats
    features.update(compute_flow_magnitude_stats(flow))

    # Multiscale entropy (stored as separate columns)
    mse = compute_multiscale_entropy(ts_mean)
    for i, val in enumerate(mse, start=1):
        features[f"mse_scale_{i}"] = float(val)

    return features


def extract_all_features(
    flow_dir: Path,
    output_dir: Path,
    groups: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Extract features from all cached flow .npy files.

    Args:
        flow_dir: Directory containing {group}/{video}.npy flow files.
        output_dir: Where to save CSVs.
        groups: If given, only process these groups.

    Returns:
        Combined DataFrame with one row per video.
    """
    flow_dir = Path(flow_dir)
    df_dir = Path(output_dir) / "dataframes"
    df_dir.mkdir(parents=True, exist_ok=True)

    if groups is None:
        groups = sorted(
            d.name for d in flow_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    all_rows: List[Dict] = []

    for group in groups:
        group_dir = flow_dir / group
        if not group_dir.exists():
            logger.warning("Flow directory not found: %s", group_dir)
            continue

        npy_files = sorted(group_dir.glob("*.npy"))
        # Filter out fps files
        npy_files = [f for f in npy_files if not f.stem.endswith("_fps")]

        if not npy_files:
            logger.warning("No flow files in %s", group_dir)
            continue

        logger.info("Extracting features for group '%s': %d videos", group, len(npy_files))

        for npy_path in npy_files:
            fps_path = npy_path.parent / f"{npy_path.stem}_fps.npy"
            fps = float(np.load(fps_path)) if fps_path.exists() else 30.0

            flow = np.load(npy_path)
            logger.info("  %s: %s", npy_path.stem, flow.shape)

            try:
                row = extract_features_single(flow, fps, npy_path.stem, group)
                all_rows.append(row)
            except Exception:
                logger.exception("  FAILED: %s", npy_path.stem)

    if not all_rows:
        logger.warning("No features extracted.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Save per-group and combined
    for group in df["group"].unique():
        group_df = df[df["group"] == group]
        group_df.to_csv(df_dir / f"{group}_features.csv", index=False)

    combined_path = df_dir / "all_features.csv"
    df.to_csv(combined_path, index=False)
    logger.info("Saved features to %s (%d rows)", combined_path, len(df))

    return df


def extract_clip_features(
    clips: List[np.ndarray],
    labels: List[str],
    video_ids: List[str],
    fps_values: List[float],
) -> pd.DataFrame:
    """Extract features from pre-extracted clips.

    Wraps extract_features_single for clip-level operation, adding
    a video_id column for grouped cross-validation.

    Args:
        clips: List of [N, H, W, 2] flow clips.
        labels: Group label per clip.
        video_ids: Source video identifier per clip.
        fps_values: FPS per clip.

    Returns:
        DataFrame with one row per clip, including 'video_id' column.
    """
    rows: List[Dict] = []
    for i, (clip, label, vid_id, fps) in enumerate(
        zip(clips, labels, video_ids, fps_values)
    ):
        try:
            row = extract_features_single(clip, fps, f"{vid_id}_clip{i}", label)
            row["video_id"] = vid_id
            rows.append(row)
        except Exception:
            logger.exception("Failed on clip %d from %s", i, vid_id)

    return pd.DataFrame(rows) if rows else pd.DataFrame()
