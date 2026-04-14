"""Feature extraction from dense optical flow fields.

Thin wrapper over `pipeline.compact` and `pipeline.feature_battery`:
the compact module turns a full-res flow field into its mag_ts + spatial
summaries in-memory, then the feature battery computes all ~100+ scalar
features from those summaries.

The canonical entry point for downstream clients is:
  * `extract_features_single(flow, fps, video_name, group)` — one video/clip
  * `extract_all_features(flow_dir, output_dir, groups=...)` — batch over a
    `flow_dir/{group}/*.npy` tree, writes per-group + combined CSVs
  * `extract_clip_features(clips, labels, video_ids, fps_values)` — clip-level
    batch with a `video_id` column for grouped CV

For videos processed with the streaming H100 path, compact arrays already
live on disk under `output/compact/{group}/{stem}_{mag_ts,spatial}.npy`;
call `compute_all_features` from `pipeline.feature_battery` directly rather
than re-deriving from a full-res flow.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from pipeline.compact import (
    compute_magnitude_timeseries,
    compute_spatial_summary,
)
from pipeline.feature_battery import compute_all_features

logger = logging.getLogger(__name__)


def extract_features_single(
    flow: np.ndarray, fps: float, video_name: str, group: str
) -> Dict[str, float | str]:
    """Extract all features from a single flow array.

    Args:
        flow: [N, H, W, 2] dense optical flow.
        fps: Frame rate.
        video_name: Identifier for this video/clip.
        group: Class label.

    Returns:
        Row dict with metadata + ~100 scalar features.
    """
    mag_ts = compute_magnitude_timeseries(flow)
    spatial = compute_spatial_summary(flow)
    return compute_all_features(
        mag_ts=mag_ts,
        spatial=spatial,
        fps=fps,
        video_name=video_name,
        group=group,
    )


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
        npy_files = [f for f in npy_files if not f.stem.endswith("_fps")]

        if not npy_files:
            logger.warning("No flow files in %s", group_dir)
            continue

        logger.info(
            "Extracting features for group '%s': %d videos",
            group, len(npy_files),
        )

        for npy_path in npy_files:
            fps_path = npy_path.parent / f"{npy_path.stem}_fps.npy"
            fps = float(np.load(fps_path)) if fps_path.exists() else 30.0

            flow = np.load(npy_path, mmap_mode="r")
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

    Adds a ``video_id`` column for grouped cross-validation.

    Args:
        clips: List of [N, H, W, 2] flow clips.
        labels: Group label per clip.
        video_ids: Source video identifier per clip.
        fps_values: FPS per clip.

    Returns:
        DataFrame with one row per clip, including 'video_id'.
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
