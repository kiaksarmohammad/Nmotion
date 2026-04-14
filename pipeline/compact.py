"""Compact representations of dense optical flow.

Converts [N, H, W, 2] flow fields into transferable summaries (~75 MB/video
vs ~10 GB full-res) that preserve enough information for downstream feature
discovery:

  1. Magnitude time series (6 stats/frame): mean, max, std, median, p5, p95
  2. Spatial summary (12 features/frame):   quadrants, symmetry, curl,
                                            divergence, coherence, and
                                            directional flow differences
  3. Downscaled flow (128x128, float16):    spatial patterns at body scale

The same functions are used by the batched H100 extractor (streaming) and
by the post-hoc converter (reads full-res .npy via memmap).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def compute_magnitude_timeseries(flow: np.ndarray) -> np.ndarray:
    """Per-frame flow magnitude statistics.

    Args:
        flow: [N, H, W, 2] flow field (memmap-compatible).

    Returns:
        [N, 6] float32 array: mean, max, std, median, p5, p95 of magnitude per frame.
    """
    n_frames = flow.shape[0]
    stats = np.empty((n_frames, 6), dtype=np.float32)

    for i in range(n_frames):
        mag = np.sqrt(flow[i, :, :, 0] ** 2 + flow[i, :, :, 1] ** 2)
        flat = mag.ravel()
        stats[i, 0] = flat.mean()
        stats[i, 1] = flat.max()
        stats[i, 2] = flat.std()
        stats[i, 3] = np.median(flat)
        stats[i, 4] = np.percentile(flat, 5)
        stats[i, 5] = np.percentile(flat, 95)

    return stats


def compute_spatial_summary(flow: np.ndarray) -> np.ndarray:
    """Per-frame spatial features from flow field.

    Args:
        flow: [N, H, W, 2] flow field.

    Returns:
        [N, 12] float32 array per frame:
          [0:4]  — quadrant mean magnitudes (TL, TR, BL, BR)
          [4]    — symmetry index (left/right energy ratio)
          [5]    — mean curl (rotation)
          [6]    — mean divergence (expansion/contraction)
          [7]    — spatial coherence (mean alignment of flow vectors)
          [8:12] — quadrant mean u, v (directional flow per quadrant)
    """
    n_frames, h, w, _ = flow.shape
    mid_h, mid_w = h // 2, w // 2
    summary = np.empty((n_frames, 12), dtype=np.float32)

    for i in range(n_frames):
        u, v = flow[i, :, :, 0], flow[i, :, :, 1]
        mag = np.sqrt(u ** 2 + v ** 2)

        # Quadrant magnitudes
        summary[i, 0] = mag[:mid_h, :mid_w].mean()  # TL
        summary[i, 1] = mag[:mid_h, mid_w:].mean()   # TR
        summary[i, 2] = mag[mid_h:, :mid_w].mean()   # BL
        summary[i, 3] = mag[mid_h:, mid_w:].mean()   # BR

        # Symmetry: ratio of left vs right kinetic energy
        left_energy = (mag[:, :mid_w] ** 2).sum()
        right_energy = (mag[:, mid_w:] ** 2).sum()
        total = left_energy + right_energy
        summary[i, 4] = left_energy / total if total > 0 else 0.5

        # Curl: dv/dx - du/dy (rotation in the flow field)
        dvdx = np.gradient(v, axis=1)
        dudy = np.gradient(u, axis=0)
        summary[i, 5] = (dvdx - dudy).mean()

        # Divergence: du/dx + dv/dy (expansion/contraction)
        dudx = np.gradient(u, axis=1)
        dvdy = np.gradient(v, axis=0)
        summary[i, 6] = (dudx + dvdy).mean()

        # Spatial coherence: how aligned are flow vectors?
        # Mean of unit vectors dotted with mean direction
        mean_u, mean_v = u.mean(), v.mean()
        mean_mag = np.sqrt(mean_u ** 2 + mean_v ** 2)
        if mean_mag > 1e-6:
            dot = (u * mean_u + v * mean_v) / (mag * mean_mag + 1e-8)
            summary[i, 7] = dot.mean()
        else:
            summary[i, 7] = 0.0

        # Directional flow per quadrant (mean u, v compressed into 4 values)
        summary[i, 8] = u[:mid_h, :].mean() - u[mid_h:, :].mean()  # top-bottom u
        summary[i, 9] = v[:mid_h, :].mean() - v[mid_h:, :].mean()  # top-bottom v
        summary[i, 10] = u[:, :mid_w].mean() - u[:, mid_w:].mean()  # left-right u
        summary[i, 11] = v[:, :mid_w].mean() - v[:, mid_w:].mean()  # left-right v

    return summary


def downscale_flow(flow: np.ndarray, target_size: int = 128) -> np.ndarray:
    """Downscale flow field to target_size x target_size, float16.

    Scales the displacement vectors proportionally to the spatial resize.

    Args:
        flow: [N, H, W, 2] full-resolution flow.
        target_size: Output spatial dimension.

    Returns:
        [N, target_size, target_size, 2] float16 flow.
    """
    n_frames, h, w, _ = flow.shape
    scale_x = target_size / w
    scale_y = target_size / h
    out = np.empty((n_frames, target_size, target_size, 2), dtype=np.float16)

    for i in range(n_frames):
        resized = cv2.resize(flow[i], (target_size, target_size),
                             interpolation=cv2.INTER_AREA)
        resized[:, :, 0] *= scale_x  # scale u displacement
        resized[:, :, 1] *= scale_y  # scale v displacement
        out[i] = resized.astype(np.float16)

    return out


def save_compact_representation(
    flow_path: Path,
    output_dir: Path,
    stem: str,
    target_size: int = 128,
) -> None:
    """Compute and save compact representation from a full-resolution flow file.

    Args:
        flow_path: Path to [N, H, W, 2] float32 .npy flow file.
        output_dir: Directory to save compact files.
        stem: Video stem name.
        target_size: Spatial size for downscaled flow.
    """
    import logging
    logger = logging.getLogger(__name__)

    output_dir.mkdir(parents=True, exist_ok=True)

    flow = np.load(str(flow_path), mmap_mode="r")
    logger.info("  compact: %s — %s", stem, flow.shape)

    mag_ts = compute_magnitude_timeseries(flow)
    np.save(output_dir / f"{stem}_mag_ts.npy", mag_ts)

    spatial = compute_spatial_summary(flow)
    np.save(output_dir / f"{stem}_spatial.npy", spatial)

    small_flow = downscale_flow(flow, target_size)
    np.save(output_dir / f"{stem}_flow128.npy", small_flow)

    size_mb = (mag_ts.nbytes + spatial.nbytes + small_flow.nbytes) / 1e6
    logger.info("  compact saved: %.1f MB", size_mb)
