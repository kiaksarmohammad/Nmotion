"""
Exhaustive feature battery for neonatal movement classification.

Computes ~100+ candidate features from compact flow representations:
  - mag_ts:  [N, 6] per-frame magnitude stats (mean, max, std, median, p5, p95)
  - spatial: [N, 12] per-frame spatial summaries
  - flow128: [N, 128, 128, 2] downscaled flow (float16)
  - fps:     scalar frame rate

Features are organized by category. Each returns a dict of {name: scalar}.
NaN is returned for features that can't be computed (e.g., too few frames).
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict

import antropy
try:
    import nolds
except Exception:
    nolds = None
import numpy as np
from scipy import stats as sp_stats
from scipy.signal import welch

logger = logging.getLogger(__name__)

# Suppress numerical warnings from edge cases (short series, flat signals)
warnings.filterwarnings("ignore", category=RuntimeWarning)

NAN = float("nan")


def _safe(fn, *args, **kwargs) -> float:
    """Call fn, return NaN on any exception."""
    try:
        v = float(fn(*args, **kwargs))
        return v if np.isfinite(v) else NAN
    except Exception:
        return NAN


# ═══════════════════════════════════════════════════════════════════════════
# Category 1: Entropy & Complexity
# ═══════════════════════════════════════════════════════════════════════════

def entropy_features(ts: np.ndarray) -> Dict[str, float]:
    """Entropy and complexity measures from a 1D time series."""
    out: Dict[str, float] = {}

    out["sample_entropy"] = _safe(antropy.sample_entropy, ts, order=2)
    out["spectral_entropy"] = _safe(
        antropy.spectral_entropy, ts, sf=1.0, method="welch", normalize=True,
    )
    out["permutation_entropy"] = _safe(
        antropy.perm_entropy, ts, order=3, normalize=True,
    )
    out["svd_entropy"] = _safe(
        antropy.svd_entropy, ts, order=3, normalize=True,
    )
    out["approximate_entropy"] = _safe(antropy.app_entropy, ts, order=2)

    # Lempel-Ziv complexity (binarize at median)
    binary = (ts > np.median(ts)).astype(int)
    out["lempel_ziv"] = _safe(antropy.lziv_complexity, binary, normalize=True)

    # Multiscale entropy at key scales: 1, 5, 10, 15, 20
    for scale in [1, 5, 10, 15, 20]:
        if scale == 1:
            coarsened = ts
        else:
            n = len(ts) - (len(ts) % scale)
            if n < 20 * scale:
                out[f"mse_scale_{scale}"] = NAN
                continue
            coarsened = ts[:n].reshape(-1, scale).mean(axis=1)
        out[f"mse_scale_{scale}"] = _safe(antropy.sample_entropy, coarsened, order=2)

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Category 2: Fractal & Long-Range Dependence
# ═══════════════════════════════════════════════════════════════════════════

def fractal_features(ts: np.ndarray) -> Dict[str, float]:
    """Fractal and long-range dependence measures."""
    out: Dict[str, float] = {}

    out["dfa_alpha"] = _safe(antropy.detrended_fluctuation, ts)
    out["hurst_rs"] = _safe(nolds.hurst_rs, ts) if nolds is not None else np.nan
    out["higuchi_fd"] = _safe(antropy.higuchi_fd, ts)
    out["katz_fd"] = _safe(antropy.katz_fd, ts)
    out["petrosian_fd"] = _safe(antropy.petrosian_fd, ts)

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Category 3: Frequency Domain
# ═══════════════════════════════════════════════════════════════════════════

def frequency_features(ts: np.ndarray, fps: float) -> Dict[str, float]:
    """Spectral features from power spectral density."""
    out: Dict[str, float] = {}

    nperseg = min(256, len(ts))
    if nperseg < 16:
        return {k: NAN for k in [
            "peak_frequency", "spectral_centroid", "spectral_bandwidth",
            "spectral_rolloff", "spectral_flatness", "spectral_edge_95",
            "band_delta", "band_seizure", "band_normal", "band_fast",
            "band_ratio_seizure_normal",
        ]}

    freqs, psd = welch(ts, fs=fps, nperseg=nperseg)
    psd_norm = psd / (psd.sum() + 1e-12)

    out["peak_frequency"] = float(freqs[np.argmax(psd)])

    # Spectral centroid (mean frequency weighted by power)
    out["spectral_centroid"] = float(np.sum(freqs * psd_norm))

    # Spectral bandwidth (std of frequency weighted by power)
    centroid = out["spectral_centroid"]
    out["spectral_bandwidth"] = float(
        np.sqrt(np.sum(((freqs - centroid) ** 2) * psd_norm))
    )

    # Spectral rolloff (freq below which 85% of power lies)
    cumpower = np.cumsum(psd_norm)
    idx85 = np.searchsorted(cumpower, 0.85)
    out["spectral_rolloff"] = float(freqs[min(idx85, len(freqs) - 1)])

    # Spectral flatness (Wiener entropy) — 1=white noise, 0=pure tone
    log_psd = np.log(psd + 1e-12)
    geo_mean = np.exp(log_psd.mean())
    arith_mean = psd.mean()
    out["spectral_flatness"] = float(geo_mean / (arith_mean + 1e-12))

    # Spectral edge frequency (95% power cutoff)
    idx95 = np.searchsorted(cumpower, 0.95)
    out["spectral_edge_95"] = float(freqs[min(idx95, len(freqs) - 1)])

    # Band powers: delta (0-1 Hz), seizure (1-3 Hz), normal (3-8 Hz), fast (8+ Hz)
    def band_power(f_low: float, f_high: float) -> float:
        mask = (freqs >= f_low) & (freqs < f_high)
        return float(psd[mask].sum()) if mask.any() else 0.0

    total_power = psd.sum() + 1e-12
    out["band_delta"] = band_power(0, 1) / total_power
    out["band_seizure"] = band_power(1, 3) / total_power
    out["band_normal"] = band_power(3, 8) / total_power
    out["band_fast"] = band_power(8, fps / 2) / total_power

    # Key ratio: seizure-band vs normal-band
    normal_power = band_power(3, 8)
    seizure_power = band_power(1, 3)
    out["band_ratio_seizure_normal"] = seizure_power / (normal_power + 1e-12)

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Category 4: Temporal Structure
# ═══════════════════════════════════════════════════════════════════════════

def temporal_features(ts: np.ndarray, fps: float) -> Dict[str, float]:
    """Temporal structure and dynamics features."""
    out: Dict[str, float] = {}

    n = len(ts)
    if n < 10:
        keys = [
            "autocorr_halflife", "zero_crossing_rate",
            "hjorth_activity", "hjorth_mobility", "hjorth_complexity",
            "burst_fraction", "burst_mean_duration", "burst_max_duration",
            "quiescence_fraction", "coeff_variation",
        ]
        return {k: NAN for k in keys}

    # Autocorrelation half-life (frames until autocorr drops below 0.5)
    ts_centered = ts - ts.mean()
    var = np.var(ts_centered)
    if var > 1e-12:
        autocorr = np.correlate(ts_centered, ts_centered, mode="full")
        autocorr = autocorr[n - 1:] / (var * n)
        halflife_idx = np.where(autocorr < 0.5)[0]
        out["autocorr_halflife"] = float(halflife_idx[0] / fps) if len(halflife_idx) > 0 else float(n / fps)
    else:
        out["autocorr_halflife"] = NAN

    # Zero-crossing rate
    crossings = np.diff(np.sign(ts_centered))
    out["zero_crossing_rate"] = float(np.count_nonzero(crossings) / n)

    # Hjorth parameters
    d1 = np.diff(ts)
    d2 = np.diff(d1)
    activity = float(np.var(ts))
    mobility = float(np.sqrt(np.var(d1) / (activity + 1e-12)))
    complexity_d1 = float(np.sqrt(np.var(d2) / (np.var(d1) + 1e-12)))
    out["hjorth_activity"] = activity
    out["hjorth_mobility"] = mobility
    out["hjorth_complexity"] = complexity_d1 / (mobility + 1e-12) if mobility > 1e-12 else NAN

    # Burst analysis (movement episodes above 75th percentile)
    threshold = np.percentile(ts, 75)
    above = ts > threshold
    transitions = np.diff(above.astype(int))
    burst_starts = np.where(transitions == 1)[0]
    burst_ends = np.where(transitions == -1)[0]

    if len(burst_starts) > 0 and len(burst_ends) > 0:
        if burst_ends[0] < burst_starts[0]:
            burst_ends = burst_ends[1:]
        min_len = min(len(burst_starts), len(burst_ends))
        burst_starts = burst_starts[:min_len]
        burst_ends = burst_ends[:min_len]
        durations = (burst_ends - burst_starts) / fps

        out["burst_fraction"] = float(above.sum() / n)
        out["burst_mean_duration"] = float(durations.mean()) if len(durations) > 0 else NAN
        out["burst_max_duration"] = float(durations.max()) if len(durations) > 0 else NAN
    else:
        out["burst_fraction"] = float(above.sum() / n)
        out["burst_mean_duration"] = NAN
        out["burst_max_duration"] = NAN

    # Quiescence fraction (below 10th percentile)
    threshold_low = np.percentile(ts, 10)
    out["quiescence_fraction"] = float((ts < threshold_low).sum() / n)

    # Coefficient of variation
    mean_val = ts.mean()
    out["coeff_variation"] = float(ts.std() / (mean_val + 1e-12))

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Category 5: Spatial Features (from spatial summary [N, 12])
# ═══════════════════════════════════════════════════════════════════════════

def spatial_features(spatial: np.ndarray) -> Dict[str, float]:
    """Features from per-frame spatial summaries [N, 12].

    Columns: TL, TR, BL, BR mags, symmetry, curl, div, coherence,
             u_topbot, v_topbot, u_leftright, v_leftright
    """
    out: Dict[str, float] = {}

    if spatial.shape[0] < 5:
        return {}

    # Quadrant magnitude statistics
    quads = spatial[:, :4]  # [N, 4]
    out["quad_mean"] = float(quads.mean())
    out["quad_std_across"] = float(quads.std(axis=1).mean())  # spatial variability
    out["quad_std_temporal"] = float(quads.mean(axis=1).std())  # temporal variability

    # Cross-quadrant correlation (diagonal pairs)
    tl, tr, bl, br = spatial[:, 0], spatial[:, 1], spatial[:, 2], spatial[:, 3]
    out["quad_corr_diag1"] = _safe(lambda: np.corrcoef(tl, br)[0, 1])
    out["quad_corr_diag2"] = _safe(lambda: np.corrcoef(tr, bl)[0, 1])
    out["quad_corr_lr_top"] = _safe(lambda: np.corrcoef(tl, tr)[0, 1])
    out["quad_corr_lr_bot"] = _safe(lambda: np.corrcoef(bl, br)[0, 1])

    # Symmetry index stats
    sym = spatial[:, 4]
    out["symmetry_mean"] = float(sym.mean())
    out["symmetry_std"] = float(sym.std())

    # Curl (rotation) stats
    curl = spatial[:, 5]
    out["curl_mean"] = float(curl.mean())
    out["curl_std"] = float(curl.std())
    out["curl_abs_mean"] = float(np.abs(curl).mean())

    # Divergence (expansion/contraction) stats
    div = spatial[:, 6]
    out["divergence_mean"] = float(div.mean())
    out["divergence_std"] = float(div.std())
    out["divergence_abs_mean"] = float(np.abs(div).mean())

    # Coherence stats
    coh = spatial[:, 7]
    out["coherence_mean"] = float(coh.mean())
    out["coherence_std"] = float(coh.std())

    # Directional flow differences
    out["flow_diff_ud_u_mean"] = float(np.abs(spatial[:, 8]).mean())
    out["flow_diff_ud_v_mean"] = float(np.abs(spatial[:, 9]).mean())
    out["flow_diff_lr_u_mean"] = float(np.abs(spatial[:, 10]).mean())
    out["flow_diff_lr_v_mean"] = float(np.abs(spatial[:, 11]).mean())

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Category 6: Kinematic Features
# ═══════════════════════════════════════════════════════════════════════════

def kinematic_features(ts: np.ndarray, fps: float) -> Dict[str, float]:
    """Motion quality and kinematic features."""
    out: Dict[str, float] = {}

    if len(ts) < 10:
        return {}

    # Basic magnitude stats
    out["flow_mean"] = float(ts.mean())
    out["flow_std"] = float(ts.std())
    out["flow_skew"] = _safe(sp_stats.skew, ts)
    out["flow_kurtosis"] = _safe(sp_stats.kurtosis, ts)
    out["flow_median"] = float(np.median(ts))
    out["flow_iqr"] = float(np.percentile(ts, 75) - np.percentile(ts, 25))
    out["flow_p95"] = float(np.percentile(ts, 95))
    out["flow_p5"] = float(np.percentile(ts, 5))
    out["flow_range"] = float(ts.max() - ts.min())

    # Kinetic energy (proportional to magnitude squared)
    ke = ts ** 2
    out["ke_mean"] = float(ke.mean())
    out["ke_std"] = float(ke.std())

    # Jerk (3rd temporal derivative of position → 1st derivative of magnitude)
    velocity = np.diff(ts) * fps
    acceleration = np.diff(velocity) * fps
    jerk = np.diff(acceleration) * fps

    if len(jerk) > 0:
        out["jerk_mean"] = float(np.abs(jerk).mean())
        out["jerk_std"] = float(jerk.std())
        out["jerk_max"] = float(np.abs(jerk).max())
    else:
        out["jerk_mean"] = NAN
        out["jerk_std"] = NAN
        out["jerk_max"] = NAN

    # Smoothness — spectral arc length (lower = smoother)
    # Simplified: negative of log magnitude spectrum arc length
    n = len(ts)
    if n >= 16:
        fft_mag = np.abs(np.fft.rfft(ts - ts.mean()))
        fft_mag = fft_mag / (fft_mag.max() + 1e-12)
        arc_length = np.sum(np.sqrt(1 + np.diff(fft_mag) ** 2))
        out["spectral_arc_length"] = -float(arc_length)
    else:
        out["spectral_arc_length"] = NAN

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Category 7: Recurrence Quantification (lightweight, no pyrqa)
# ═══════════════════════════════════════════════════════════════════════════

def recurrence_features(ts: np.ndarray, max_samples: int = 500) -> Dict[str, float]:
    """Lightweight recurrence quantification from 1D time series.

    Uses a distance matrix approach (no external RQA library needed).
    Subsamples to max_samples for speed.
    """
    out: Dict[str, float] = {}

    n = len(ts)
    if n < 30:
        return {
            "recurrence_rate": NAN, "determinism": NAN,
            "laminarity": NAN, "trapping_time": NAN,
            "max_diagonal": NAN,
        }

    # Subsample for speed
    if n > max_samples:
        idx = np.linspace(0, n - 1, max_samples, dtype=int)
        ts_sub = ts[idx]
    else:
        ts_sub = ts

    m = len(ts_sub)
    # Threshold at 10% of std
    eps = 0.1 * ts_sub.std()
    if eps < 1e-12:
        return {
            "recurrence_rate": NAN, "determinism": NAN,
            "laminarity": NAN, "trapping_time": NAN,
            "max_diagonal": NAN,
        }

    # Distance matrix (pairwise absolute differences)
    dist = np.abs(ts_sub[:, None] - ts_sub[None, :])
    recmat = (dist < eps).astype(np.int8)

    total_points = m * m
    recurrent = recmat.sum()
    out["recurrence_rate"] = float(recurrent / total_points)

    # Diagonal lines (determinism)
    diag_lengths = []
    for k in range(-m + 1, m):
        diag = np.diag(recmat, k)
        runs = np.diff(np.concatenate([[0], diag, [0]]))
        starts = np.where(runs == 1)[0]
        ends = np.where(runs == -1)[0]
        for s, e in zip(starts, ends):
            length = e - s
            if length >= 2:
                diag_lengths.append(length)

    if diag_lengths:
        diag_lengths = np.array(diag_lengths)
        det_points = diag_lengths.sum()
        out["determinism"] = float(det_points / (recurrent + 1e-12))
        out["max_diagonal"] = float(diag_lengths.max())
    else:
        out["determinism"] = 0.0
        out["max_diagonal"] = 0.0

    # Vertical lines (laminarity)
    vert_lengths = []
    for col in range(m):
        runs = np.diff(np.concatenate([[0], recmat[:, col], [0]]))
        starts = np.where(runs == 1)[0]
        ends = np.where(runs == -1)[0]
        for s, e in zip(starts, ends):
            length = e - s
            if length >= 2:
                vert_lengths.append(length)

    if vert_lengths:
        vert_lengths = np.array(vert_lengths)
        lam_points = vert_lengths.sum()
        out["laminarity"] = float(lam_points / (recurrent + 1e-12))
        out["trapping_time"] = float(vert_lengths.mean())
    else:
        out["laminarity"] = 0.0
        out["trapping_time"] = 0.0

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Category 8: Multi-signal features (cross-channel from mag_ts [N, 6])
# ═══════════════════════════════════════════════════════════════════════════

def multisignal_features(mag_ts: np.ndarray) -> Dict[str, float]:
    """Features comparing different magnitude statistics across time.

    mag_ts columns: [mean, max, std, median, p5, p95]
    """
    out: Dict[str, float] = {}

    if mag_ts.shape[0] < 10:
        return {}

    mean_ts, max_ts, std_ts = mag_ts[:, 0], mag_ts[:, 1], mag_ts[:, 2]
    median_ts, p5_ts, p95_ts = mag_ts[:, 3], mag_ts[:, 4], mag_ts[:, 5]

    # Dynamic range per frame (p95 - p5)
    dyn_range = p95_ts - p5_ts
    out["dynamic_range_mean"] = float(dyn_range.mean())
    out["dynamic_range_std"] = float(dyn_range.std())

    # Max/mean ratio — peakiness of flow distribution
    out["peakiness_mean"] = float((max_ts / (mean_ts + 1e-8)).mean())
    out["peakiness_std"] = float((max_ts / (mean_ts + 1e-8)).std())

    # Std/mean ratio — spatial heterogeneity over time
    out["spatial_cv_mean"] = float((std_ts / (mean_ts + 1e-8)).mean())
    out["spatial_cv_std"] = float((std_ts / (mean_ts + 1e-8)).std())

    # Correlation between mean and std (coupled vs decoupled motion)
    out["mean_std_corr"] = _safe(lambda: np.corrcoef(mean_ts, std_ts)[0, 1])

    # Correlation between mean and max
    out["mean_max_corr"] = _safe(lambda: np.corrcoef(mean_ts, max_ts)[0, 1])

    # Temporal derivative stats of mean flow
    dmean = np.diff(mean_ts)
    out["flow_acceleration_mean"] = float(np.abs(dmean).mean())
    out["flow_acceleration_std"] = float(dmean.std())

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Master function: compute all features for one video
# ═══════════════════════════════════════════════════════════════════════════

def compute_all_features(
    mag_ts: np.ndarray,
    spatial: np.ndarray,
    fps: float,
    video_name: str,
    group: str,
) -> Dict[str, float | str]:
    """Compute the full feature battery from compact representations.

    Args:
        mag_ts: [N, 6] per-frame magnitude statistics.
        spatial: [N, 12] per-frame spatial summaries.
        fps: Frame rate.
        video_name: Video identifier.
        group: Class label.

    Returns:
        Dict with ~100+ feature values plus metadata.
    """
    features: Dict[str, float | str] = {
        "video": video_name,
        "group": group,
        "n_frames": float(mag_ts.shape[0]),
        "fps": fps,
    }

    # Primary time series: per-frame mean flow magnitude
    ts_mean = mag_ts[:, 0].copy()

    # Ensure float64 for numerical stability
    ts_mean = ts_mean.astype(np.float64)

    features.update(entropy_features(ts_mean))
    features.update(fractal_features(ts_mean))
    features.update(frequency_features(ts_mean, fps))
    features.update(temporal_features(ts_mean, fps))
    features.update(spatial_features(spatial))
    features.update(kinematic_features(ts_mean, fps))
    features.update(recurrence_features(ts_mean))
    features.update(multisignal_features(mag_ts))

    return features
