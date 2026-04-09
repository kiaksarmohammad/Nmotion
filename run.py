"""
Nmotion — Neonatal movement analysis pipeline.

Usage:
    python run.py --video-dir data/videos
    python run.py --video-dir data/videos --skip-flow     # use cached .npy
    python run.py --video-dir data/videos --groups normal spasms
    python run.py --video-dir data/videos --device cpu
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from pipeline.flow_extract import extract_all_flows
from pipeline.features import extract_all_features
from pipeline.visualize import generate_figures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nmotion")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nmotion: neonatal movement analysis")
    p.add_argument(
        "--video-dir", type=Path, required=True,
        help="Root directory containing group subdirs with videos",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("output"),
        help="Output directory (default: output/)",
    )
    p.add_argument(
        "--skip-flow", action="store_true",
        help="Skip flow extraction, use cached .npy files",
    )
    p.add_argument(
        "--groups", nargs="+", default=None,
        help="Subset of groups to process (default: all subdirs)",
    )
    p.add_argument(
        "--device", default=None,
        help="Torch device: 'cuda' or 'cpu' (default: auto-detect)",
    )
    return p.parse_args()


def _print_summary(df: pd.DataFrame) -> None:
    """Print a summary table of key features per group."""
    if df.empty:
        logger.warning("No data to summarize.")
        return

    key_cols = [
        "sample_entropy", "spectral_entropy", "dfa_alpha",
        "symmetry_index", "peak_frequency", "kinetic_energy_mean",
        "flow_mean", "flow_std",
    ]
    present = [c for c in key_cols if c in df.columns]

    print("\n" + "=" * 70)
    print("FEATURE SUMMARY BY GROUP")
    print("=" * 70)

    summary = df.groupby("group")[present].agg(["mean", "std"])
    # Flatten multi-level columns
    summary.columns = [f"{feat} ({stat})" for feat, stat in summary.columns]

    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(summary.round(4).to_string())

    print("=" * 70)
    print(f"Total videos: {len(df)}")
    for group in df["group"].unique():
        print(f"  {group}: {len(df[df['group'] == group])}")
    print()


def main() -> None:
    args = parse_args()
    video_dir = args.video_dir
    output_dir = args.output_dir
    flow_dir = output_dir / "flows"

    if args.device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logger.info("Device: %s", device)
    logger.info("Video dir: %s", video_dir)
    logger.info("Output dir: %s", output_dir)

    # Stage 1: Flow extraction
    if not args.skip_flow:
        logger.info("=" * 50)
        logger.info("STAGE 1: OPTICAL FLOW EXTRACTION")
        logger.info("=" * 50)
        extract_all_flows(video_dir, output_dir, device=device, groups=args.groups)
    else:
        logger.info("Skipping flow extraction (--skip-flow)")

    # Stage 2: Feature extraction
    logger.info("=" * 50)
    logger.info("STAGE 2: FEATURE EXTRACTION")
    logger.info("=" * 50)
    df = extract_all_features(flow_dir, output_dir, groups=args.groups)

    if df.empty:
        logger.error("No features extracted. Add videos to %s/{group}/ and re-run.", video_dir)
        sys.exit(1)

    # Stage 3: Visualization
    logger.info("=" * 50)
    logger.info("STAGE 3: VISUALIZATION")
    logger.info("=" * 50)
    generate_figures(df, flow_dir, output_dir, groups=args.groups)

    _print_summary(df)
    logger.info("Done. Figures in %s/figures/, data in %s/dataframes/", output_dir, output_dir)


if __name__ == "__main__":
    main()
