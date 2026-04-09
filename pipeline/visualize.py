"""
Matplotlib figure generation for neonatal movement group comparison.

Produces 5 publication-quality figures from the feature DataFrame and
raw flow data. All saved as PNG (300 dpi) + PDF.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.signal import welch

logger = logging.getLogger(__name__)

# Consistent colors per group
GROUP_COLORS = {
    "normal": "#2ecc71",
    "spasms": "#e74c3c",
    "hypertonia": "#3498db",
}
GROUP_ORDER = ["normal", "spasms", "hypertonia"]


def _get_color(group: str) -> str:
    return GROUP_COLORS.get(group, "#95a5a6")


def _save_fig(fig: plt.Figure, output_dir: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        path = output_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", name)


def _groups_in_order(df: pd.DataFrame) -> List[str]:
    """Return groups present in df, in canonical order."""
    present = set(df["group"].unique())
    return [g for g in GROUP_ORDER if g in present] + sorted(present - set(GROUP_ORDER))


# ---------------------------------------------------------------------------
# Figure 1: KDE of flow magnitude
# ---------------------------------------------------------------------------

def plot_flow_magnitude_kde(
    flow_dir: Path, groups: List[str], output_dir: Path
) -> None:
    """Kernel density estimate of per-pixel flow magnitude, one curve per group."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for group in groups:
        group_dir = flow_dir / group
        if not group_dir.exists():
            continue

        magnitudes = []
        for npy_path in sorted(group_dir.glob("*.npy")):
            if npy_path.stem.endswith("_fps"):
                continue
            flow = np.load(npy_path)
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).ravel()
            # Subsample for KDE performance
            if len(mag) > 500_000:
                rng = np.random.default_rng(42)
                mag = rng.choice(mag, 500_000, replace=False)
            magnitudes.append(mag)

        if not magnitudes:
            continue

        all_mag = np.concatenate(magnitudes)
        # Clip extreme outliers for cleaner KDE
        p99 = np.percentile(all_mag, 99)
        all_mag = all_mag[all_mag <= p99]

        try:
            kde = sp_stats.gaussian_kde(all_mag)
            x = np.linspace(0, p99, 500)
            ax.plot(x, kde(x), label=group, color=_get_color(group), linewidth=2)
            ax.fill_between(x, kde(x), alpha=0.15, color=_get_color(group))
        except Exception:
            logger.warning("KDE failed for group %s", group)

    ax.set_xlabel("Flow Magnitude (pixels/frame)")
    ax.set_ylabel("Density")
    ax.set_title("Flow Magnitude Distribution by Group")
    ax.legend()
    ax.set_xlim(left=0)

    _save_fig(fig, output_dir, "01_flow_magnitude_kde")


# ---------------------------------------------------------------------------
# Figure 2: Multiscale entropy profiles
# ---------------------------------------------------------------------------

def plot_multiscale_entropy(df: pd.DataFrame, output_dir: Path) -> None:
    """Entropy vs. scale per group with SEM error bands."""
    mse_cols = [c for c in df.columns if c.startswith("mse_scale_")]
    scales = list(range(1, len(mse_cols) + 1))
    groups = _groups_in_order(df)

    fig, ax = plt.subplots(figsize=(8, 5))

    for group in groups:
        gdf = df[df["group"] == group][mse_cols].values  # [n_videos, n_scales]
        mean = np.nanmean(gdf, axis=0)
        sem = np.nanstd(gdf, axis=0) / np.sqrt(np.sum(~np.isnan(gdf), axis=0).clip(1))

        color = _get_color(group)
        ax.plot(scales, mean, label=group, color=color, linewidth=2, marker="o", markersize=3)
        ax.fill_between(scales, mean - sem, mean + sem, alpha=0.2, color=color)

    ax.set_xlabel("Scale")
    ax.set_ylabel("Sample Entropy")
    ax.set_title("Multiscale Entropy Profiles")
    ax.legend()
    ax.set_xticks(scales[::2])

    _save_fig(fig, output_dir, "02_multiscale_entropy")


# ---------------------------------------------------------------------------
# Figure 3: Power spectral density
# ---------------------------------------------------------------------------

def plot_psd(flow_dir: Path, groups: List[str], output_dir: Path) -> None:
    """Log-scale PSD averaged per group. Annotates seizure frequency bands."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for group in groups:
        group_dir = flow_dir / group
        if not group_dir.exists():
            continue

        all_psd = []
        common_freqs = None

        for npy_path in sorted(group_dir.glob("*.npy")):
            if npy_path.stem.endswith("_fps"):
                continue

            flow = np.load(npy_path)
            fps_path = npy_path.parent / f"{npy_path.stem}_fps.npy"
            fps = float(np.load(fps_path)) if fps_path.exists() else 30.0

            mag_mean = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean(axis=(1, 2))
            nperseg = min(256, len(mag_mean))
            freqs, psd = welch(mag_mean, fs=fps, nperseg=nperseg)

            if common_freqs is None:
                common_freqs = freqs
                all_psd.append(psd)
            elif len(freqs) == len(common_freqs):
                all_psd.append(psd)

        if not all_psd:
            continue

        mean_psd = np.mean(all_psd, axis=0)
        color = _get_color(group)
        ax.semilogy(common_freqs, mean_psd, label=group, color=color, linewidth=2)

    # Annotate seizure frequency bands
    ax.axvspan(1.0, 3.0, alpha=0.08, color="red", label="Clonic (1-3 Hz)")
    ax.axvspan(0.5, 1.0, alpha=0.08, color="orange", label="Tonic-clonic (0.5-1 Hz)")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density")
    ax.set_title("Motion Power Spectrum by Group")
    ax.legend(fontsize=8)

    _save_fig(fig, output_dir, "03_power_spectral_density")


# ---------------------------------------------------------------------------
# Figure 4: Phase space (velocity vs acceleration)
# ---------------------------------------------------------------------------

def plot_phase_space(flow_dir: Path, groups: List[str], output_dir: Path) -> None:
    """Velocity vs. acceleration scatter, colored by group.

    Seizures → limit cycles, normal → diffuse cloud, hypotonic → tight cluster.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    for group in groups:
        group_dir = flow_dir / group
        if not group_dir.exists():
            continue

        velocities, accelerations = [], []

        for npy_path in sorted(group_dir.glob("*.npy")):
            if npy_path.stem.endswith("_fps"):
                continue

            flow = np.load(npy_path)
            mag_mean = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean(axis=(1, 2))

            vel = mag_mean  # flow magnitude ≈ velocity
            acc = np.diff(vel)  # first difference ≈ acceleration

            # Subsample for plot readability
            n = min(2000, len(acc))
            rng = np.random.default_rng(42)
            idx = rng.choice(len(acc), n, replace=len(acc) < n)

            velocities.append(vel[:-1][idx])
            accelerations.append(acc[idx])

        if not velocities:
            continue

        v = np.concatenate(velocities)
        a = np.concatenate(accelerations)
        color = _get_color(group)
        ax.scatter(v, a, s=2, alpha=0.3, color=color, label=group, rasterized=True)

    ax.set_xlabel("Velocity (flow magnitude)")
    ax.set_ylabel("Acceleration (Δ velocity)")
    ax.set_title("Phase Space: Velocity vs. Acceleration")
    ax.legend(markerscale=5)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    _save_fig(fig, output_dir, "04_phase_space")


# ---------------------------------------------------------------------------
# Figure 5: DFA log-log
# ---------------------------------------------------------------------------

def plot_dfa(df: pd.DataFrame, output_dir: Path) -> None:
    """DFA alpha exponents per group as a bar chart with individual points."""
    groups = _groups_in_order(df)

    fig, ax = plt.subplots(figsize=(6, 5))

    positions = []
    for i, group in enumerate(groups):
        gdf = df[df["group"] == group]["dfa_alpha"].dropna()
        color = _get_color(group)

        # Individual points (jittered)
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(gdf))
        ax.scatter(
            np.full(len(gdf), i) + jitter, gdf.values,
            s=30, alpha=0.6, color=color, zorder=3,
        )

        # Mean + SEM bar
        mean = gdf.mean()
        sem = gdf.std() / np.sqrt(len(gdf)) if len(gdf) > 1 else 0
        ax.bar(i, mean, width=0.5, alpha=0.3, color=color, zorder=2)
        ax.errorbar(i, mean, yerr=sem, fmt="none", color="black", capsize=5, zorder=4)

        positions.append(i)

    ax.set_xticks(positions)
    ax.set_xticklabels(groups)
    ax.set_ylabel("DFA Exponent (α)")
    ax.set_title("Detrended Fluctuation Analysis")

    # Reference lines
    ax.axhline(0.5, color="gray", linewidth=1, linestyle="--", label="α=0.5 (white noise)")
    ax.legend(fontsize=8)

    _save_fig(fig, output_dir, "05_dfa_exponents")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_figures(
    df: pd.DataFrame,
    flow_dir: Path,
    output_dir: Path,
    groups: Optional[List[str]] = None,
) -> None:
    """Generate all 5 comparison figures.

    Args:
        df: Combined feature DataFrame (from extract_all_features).
        flow_dir: Directory containing {group}/*.npy flow files.
        output_dir: Where to save figures.
        groups: If given, only plot these groups.
    """
    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    flow_dir = Path(flow_dir)

    if groups is None:
        groups = _groups_in_order(df)

    logger.info("Generating figures for groups: %s", groups)

    # Figures that need raw flow data
    plot_flow_magnitude_kde(flow_dir, groups, fig_dir)
    plot_psd(flow_dir, groups, fig_dir)
    plot_phase_space(flow_dir, groups, fig_dir)

    # Figures from the feature DataFrame
    plot_multiscale_entropy(df, fig_dir)
    plot_dfa(df, fig_dir)

    logger.info("All figures saved to %s", fig_dir)
