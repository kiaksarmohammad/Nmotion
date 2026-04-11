"""Sliding-window temporal clip extraction from dense optical flow fields.

Extracts fixed-length clips from variable-length flow arrays [N, H, W, 2],
with configurable window size and overlap. Designed for downstream feature
extraction where clips need >=200 frames for meaningful entropy/DFA computation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 10 seconds at 30fps — long enough for MSE up to scale ~14
DEFAULT_WINDOW_SECONDS = 10.0
DEFAULT_OVERLAP = 0.5
HYPOTONIC_OVERLAP = 0.9


def extract_clips(
    flow: np.ndarray,
    window_frames: int,
    overlap: float = DEFAULT_OVERLAP,
) -> List[np.ndarray]:
    """Extract temporal clips from a single flow field using sliding window.

    Args:
        flow: [N, H, W, 2] dense optical flow.
        window_frames: Number of frames per clip.
        overlap: Fraction of window overlap (0.0 to 0.99).

    Returns:
        List of [window_frames, H, W, 2] arrays. If flow is shorter than
        window_frames, returns [flow] as a single clip (no padding).
    """
    n_frames = len(flow)

    if n_frames <= window_frames:
        return [flow]

    # round before int to avoid float truncation (e.g. 30*0.1 = 2.999... → 2)
    stride = max(1, round(window_frames * (1.0 - overlap)))
    clips = []
    for start in range(0, n_frames - window_frames + 1, stride):
        clips.append(flow[start : start + window_frames])

    return clips


def extract_all_clips(
    flow_dir: Path,
    groups: Optional[List[str]] = None,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    default_overlap: float = DEFAULT_OVERLAP,
    hypotonic_overlap: float = HYPOTONIC_OVERLAP,
) -> Tuple[List[np.ndarray], List[str], List[str], List[float]]:
    """Extract clips from all cached flow files, organized by group.

    Uses higher overlap for the hypotonic group to maximize samples
    from the critically underrepresented class.

    Args:
        flow_dir: Directory containing {group}/{video}.npy flow files.
        groups: If given, only process these groups.
        window_seconds: Clip duration in seconds.
        default_overlap: Overlap fraction for most groups.
        hypotonic_overlap: Higher overlap for hypotonic group.

    Returns:
        Tuple of (clips, labels, video_ids, fps_values) where:
        - clips: list of [window_frames, H, W, 2] arrays
        - labels: list of group name per clip
        - video_ids: list of source video stem per clip (for grouped CV)
        - fps_values: list of fps per clip
    """
    flow_dir = Path(flow_dir)

    if groups is None:
        groups = sorted(
            d.name for d in flow_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    all_clips: List[np.ndarray] = []
    all_labels: List[str] = []
    all_video_ids: List[str] = []
    all_fps: List[float] = []

    for group in groups:
        group_dir = flow_dir / group
        if not group_dir.exists():
            logger.warning("Flow directory not found: %s", group_dir)
            continue

        overlap = hypotonic_overlap if group == "hypotonic" else default_overlap

        npy_files = sorted(
            f for f in group_dir.glob("*.npy")
            if not f.stem.endswith("_fps")
        )

        logger.info(
            "Extracting clips for '%s': %d videos, overlap=%.0f%%",
            group, len(npy_files), overlap * 100,
        )

        for npy_path in npy_files:
            fps_path = npy_path.parent / f"{npy_path.stem}_fps.npy"
            fps = float(np.load(fps_path)) if fps_path.exists() else 30.0
            window_frames = max(1, int(window_seconds * fps))

            flow = np.load(npy_path)
            clips = extract_clips(flow, window_frames, overlap)

            all_clips.extend(clips)
            all_labels.extend([group] * len(clips))
            all_video_ids.extend([npy_path.stem] * len(clips))
            all_fps.extend([fps] * len(clips))

            logger.info(
                "  %s: %d frames → %d clips (window=%d, overlap=%.0f%%)",
                npy_path.stem, len(flow), len(clips), window_frames, overlap * 100,
            )

    logger.info(
        "Total clips: %d (%s)",
        len(all_clips),
        ", ".join(f"{g}: {all_labels.count(g)}" for g in groups),
    )

    return all_clips, all_labels, all_video_ids, all_fps
