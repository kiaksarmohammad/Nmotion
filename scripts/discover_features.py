#!/usr/bin/env python3
"""
Feature discovery pipeline for neonatal movement classification.

Loads compact flow representations, computes the full feature battery,
runs statistical tests (Kruskal-Wallis, Mann-Whitney U, AUROC),
and produces ranked outputs + visualizations.

Usage:
    python scripts/discover_features.py --compact-dir output/compact --output-dir output/feature_discovery
"""

from __future__ import annotations

import argparse
import logging
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

# Direct import to avoid pipeline/__init__.py pulling in torch/cv2
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "feature_battery",
    str(Path(__file__).resolve().parent.parent / "pipeline" / "feature_battery.py"),
)
_fb = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_fb)
compute_all_features = _fb.compute_all_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("discover")

VIDEO_SUFFIXES = {"_mag_ts.npy", "_spatial.npy", "_flow128.npy", "_fps.npy"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def discover_videos(compact_dir: Path) -> list[dict]:
    """Find all videos with compact representations."""
    videos = []
    for group_dir in sorted(compact_dir.iterdir()):
        if not group_dir.is_dir() or group_dir.name.startswith("."):
            continue
        group = group_dir.name

        # Find unique stems by looking for _mag_ts.npy files
        for mag_path in sorted(group_dir.glob("*_mag_ts.npy")):
            stem = mag_path.name.replace("_mag_ts.npy", "")
            spatial_path = group_dir / f"{stem}_spatial.npy"
            fps_path = group_dir / f"{stem}_fps.npy"

            if not spatial_path.exists():
                logger.warning("Missing spatial for %s/%s, skipping", group, stem)
                continue

            fps = float(np.load(fps_path)) if fps_path.exists() else 30.0

            videos.append({
                "stem": stem,
                "group": group,
                "mag_ts_path": mag_path,
                "spatial_path": spatial_path,
                "fps": fps,
            })

    return videos


def load_and_compute(videos: list[dict]) -> pd.DataFrame:
    """Load compact representations and compute feature battery."""
    rows = []
    for i, v in enumerate(videos):
        logger.info(
            "  [%d/%d] %s/%s", i + 1, len(videos), v["group"], v["stem"],
        )
        try:
            mag_ts = np.load(v["mag_ts_path"])
            spatial = np.load(v["spatial_path"])

            row = compute_all_features(
                mag_ts=mag_ts,
                spatial=spatial,
                fps=v["fps"],
                video_name=v["stem"],
                group=v["group"],
            )
            rows.append(row)
        except Exception:
            logger.exception("  FAILED: %s/%s", v["group"], v["stem"])

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_statistical_tests(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Run Kruskal-Wallis + pairwise Mann-Whitney for each feature."""
    groups = sorted(df["group"].unique())
    group_pairs = list(combinations(groups, 2))

    results = []
    for feat in feature_cols:
        vals = df[feat].dropna()
        if len(vals) < 10:
            continue

        # Get per-group values
        group_vals = {g: df.loc[df["group"] == g, feat].dropna().values for g in groups}
        non_empty = {g: v for g, v in group_vals.items() if len(v) >= 3}

        if len(non_empty) < 2:
            continue

        row = {"feature": feat}

        # Kruskal-Wallis H-test (non-parametric ANOVA)
        try:
            stat, p = sp_stats.kruskal(*non_empty.values())
            row["kw_stat"] = stat
            row["kw_pvalue"] = p
        except Exception:
            row["kw_stat"] = np.nan
            row["kw_pvalue"] = np.nan

        # Pairwise Mann-Whitney U + effect size
        for g1, g2 in group_pairs:
            if g1 not in non_empty or g2 not in non_empty:
                continue

            v1, v2 = non_empty[g1], non_empty[g2]
            pair = f"{g1}_vs_{g2}"

            try:
                stat, p = sp_stats.mannwhitneyu(v1, v2, alternative="two-sided")
                n1, n2 = len(v1), len(v2)
                # Rank-biserial correlation as effect size
                rbc = 1 - (2 * stat) / (n1 * n2)
                row[f"mw_p_{pair}"] = p
                row[f"effect_{pair}"] = abs(rbc)
            except Exception:
                row[f"mw_p_{pair}"] = np.nan
                row[f"effect_{pair}"] = np.nan

            # AUROC for this pair
            try:
                labels = np.concatenate([np.zeros(len(v1)), np.ones(len(v2))])
                scores = np.concatenate([v1, v2])
                auc = roc_auc_score(labels, scores)
                # Ensure AUC >= 0.5 (flip if needed — direction doesn't matter)
                row[f"auroc_{pair}"] = max(auc, 1 - auc)
            except Exception:
                row[f"auroc_{pair}"] = np.nan

        results.append(row)

    ranking = pd.DataFrame(results)

    # Sort by Kruskal-Wallis p-value (most significant first)
    ranking = ranking.sort_values("kw_pvalue").reset_index(drop=True)

    return ranking


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_top_features(
    df: pd.DataFrame,
    ranking: pd.DataFrame,
    output_dir: Path,
    n_top: int = 20,
) -> None:
    """Violin plots for the top-N most discriminative features."""
    groups = sorted(df["group"].unique())
    colors = {
        "normal": "#2ecc71",
        "seizure": "#e74c3c",
        "hypotonic": "#3498db",
    }
    default_colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))

    top_features = ranking.head(n_top)["feature"].tolist()
    if not top_features:
        logger.warning("No features to plot")
        return

    n_cols = 4
    n_rows = (len(top_features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).ravel()

    for i, feat in enumerate(top_features):
        ax = axes[i]
        data = [df.loc[df["group"] == g, feat].dropna().values for g in groups]

        parts = ax.violinplot(data, showmeans=True, showmedians=True)
        for j, pc in enumerate(parts["bodies"]):
            c = colors.get(groups[j], default_colors[j])
            pc.set_facecolor(c)
            pc.set_alpha(0.7)

        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels(groups, fontsize=8)
        ax.set_title(feat, fontsize=9)

        # Show p-value
        p = ranking.loc[ranking["feature"] == feat, "kw_pvalue"].values
        if len(p) > 0 and np.isfinite(p[0]):
            ax.text(0.02, 0.98, f"p={p[0]:.2e}", transform=ax.transAxes,
                    fontsize=7, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    # Hide empty axes
    for j in range(len(top_features), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Top {len(top_features)} Discriminative Features (by Kruskal-Wallis p-value)",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "top_violins.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "top_violins.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved violin plots")


def plot_correlation_matrix(
    df: pd.DataFrame,
    ranking: pd.DataFrame,
    output_dir: Path,
    n_top: int = 30,
) -> None:
    """Correlation heatmap for top features (redundancy detection)."""
    top_features = ranking.head(n_top)["feature"].tolist()
    if len(top_features) < 3:
        return

    feat_df = df[top_features].dropna(axis=1, how="all")
    corr = feat_df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(corr.columns, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature Correlation Matrix (Top 30)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved correlation matrix")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Feature discovery pipeline")
    parser.add_argument(
        "--compact-dir", type=Path, required=True,
        help="Directory with compact representations (output/compact/)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/feature_discovery"),
        help="Where to save results",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover videos
    logger.info("Discovering videos in %s", args.compact_dir)
    videos = discover_videos(args.compact_dir)
    logger.info("Found %d videos", len(videos))

    for g in sorted(set(v["group"] for v in videos)):
        count = sum(1 for v in videos if v["group"] == g)
        logger.info("  %s: %d videos", g, count)

    if not videos:
        logger.error("No videos found!")
        return

    # Compute features
    logger.info("Computing feature battery...")
    df = load_and_compute(videos)
    logger.info("Feature matrix: %d videos × %d features", len(df), len(df.columns))

    # Save raw features
    df.to_csv(output_dir / "all_features.csv", index=False)
    logger.info("Saved %s", output_dir / "all_features.csv")

    # Identify numeric feature columns (exclude metadata)
    meta_cols = {"video", "group", "n_frames", "fps"}
    feature_cols = [c for c in df.columns if c not in meta_cols and df[c].dtype != object]

    # Report NaN coverage
    nan_frac = df[feature_cols].isna().mean()
    good_features = nan_frac[nan_frac < 0.1].index.tolist()
    logger.info(
        "Features with <10%% NaN: %d/%d", len(good_features), len(feature_cols),
    )

    # Statistical tests
    logger.info("Running statistical tests on %d features...", len(good_features))
    ranking = run_statistical_tests(df, good_features)
    ranking.to_csv(output_dir / "ranking.csv", index=False)
    logger.info("Saved %s", output_dir / "ranking.csv")

    # Report top hits
    sig = ranking[ranking["kw_pvalue"] < 0.05]
    logger.info("=" * 60)
    logger.info("SIGNIFICANT FEATURES (p < 0.05): %d", len(sig))
    logger.info("=" * 60)
    for _, row in sig.head(20).iterrows():
        logger.info(
            "  %-35s  p=%.2e  effect=%.3f",
            row["feature"],
            row["kw_pvalue"],
            max(
                row.get(c, 0) for c in row.index if c.startswith("effect_")
            ) if any(c.startswith("effect_") for c in row.index) else 0,
        )

    # Visualizations
    logger.info("Generating plots...")
    plot_top_features(df, ranking, output_dir)
    plot_correlation_matrix(df, ranking, output_dir)

    logger.info("=" * 60)
    logger.info("DISCOVERY COMPLETE")
    logger.info("  Features CSV: %s", output_dir / "all_features.csv")
    logger.info("  Rankings CSV: %s", output_dir / "ranking.csv")
    logger.info("  Violin plots: %s", output_dir / "top_violins.png")
    logger.info("  Correlations: %s", output_dir / "correlation_matrix.png")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
