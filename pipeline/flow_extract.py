"""
Dense optical flow extraction using torchvision RAFT.

Extracts per-frame-pair flow fields from video files and saves as .npy arrays.
Start with torchvision RAFT-Large (zero extra deps, well-tested).
RC-1: swap to SEA-RAFT or FlowSeek if quality is insufficient on neonatal video.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def _load_frames(video_path: Path) -> tuple[np.ndarray, float]:
    """Load all frames from a video file.

    Returns:
        frames: uint8 array [N, H, W, 3] (BGR)
        fps: frames per second
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) < 2:
        raise ValueError(f"Video has {len(frames)} frames, need at least 2: {video_path}")

    return np.stack(frames, axis=0), fps


def _preprocess_pair(
    frame1: np.ndarray, frame2: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a BGR frame pair to RAFT input tensors.

    RAFT expects float32 tensors in [0, 1] range, shape [1, 3, H, W].
    Resizes to dimensions divisible by 8 (RAFT requirement).
    """
    def _to_tensor(bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        # Pad to dimensions divisible by 8
        _, h, w = t.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="constant")
        return t.unsqueeze(0).to(device)

    return _to_tensor(frame1), _to_tensor(frame2)


def extract_flow(
    video_path: Path,
    device: str = "cuda",
) -> tuple[np.ndarray, float]:
    """Extract dense optical flow from a video using RAFT-Large.

    Args:
        video_path: Path to video file.
        device: "cuda" or "cpu".

    Returns:
        flow_fields: float32 array [N-1, H, W, 2] (u, v displacement)
        fps: video frame rate
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info("Loading RAFT-Large on %s", dev)

    weights = Raft_Large_Weights.C_T_SKHT_V2
    model = raft_large(weights=weights).to(dev).eval()

    frames, fps = _load_frames(video_path)
    n_frames = len(frames)
    logger.info("Loaded %d frames (%.1f fps) from %s", n_frames, fps, video_path.name)

    flows = []
    with torch.no_grad():
        for i in range(n_frames - 1):
            t1, t2 = _preprocess_pair(frames[i], frames[i + 1], dev)
            # RAFT returns a list of flow predictions (one per iteration), take the last
            flow_predictions = model(t1, t2)
            flow = flow_predictions[-1]  # [1, 2, H, W]
            # Crop padding back to original frame size
            h, w = frames[i].shape[:2]
            flow = flow[0, :, :h, :w]  # [2, H, W]
            flows.append(flow.cpu().numpy().transpose(1, 2, 0))  # [H, W, 2]

            if (i + 1) % 100 == 0:
                logger.info("  processed %d/%d frame pairs", i + 1, n_frames - 1)

    flow_fields = np.stack(flows, axis=0)  # [N-1, H, W, 2]
    logger.info("Flow extraction complete: %s", flow_fields.shape)
    return flow_fields, fps


def extract_all_flows(
    video_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    groups: Optional[List[str]] = None,
) -> Dict[str, List[Path]]:
    """Batch-extract flow from all videos organized by group.

    Expects: video_dir/{group}/*.mp4
    Outputs: output_dir/flows/{group}/{video_stem}.npy

    Args:
        video_dir: Root directory containing group subdirs.
        output_dir: Root output directory.
        device: "cuda" or "cpu".
        groups: If given, only process these groups. Otherwise, all subdirs.

    Returns:
        Dict mapping group name → list of saved .npy paths.
    """
    video_dir = Path(video_dir)
    flow_dir = Path(output_dir) / "flows"
    flow_dir.mkdir(parents=True, exist_ok=True)

    if groups is None:
        groups = sorted(
            d.name for d in video_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    result: Dict[str, List[Path]] = {}

    for group in groups:
        group_video_dir = video_dir / group
        if not group_video_dir.exists():
            logger.warning("Group directory not found: %s", group_video_dir)
            continue

        group_flow_dir = flow_dir / group
        group_flow_dir.mkdir(parents=True, exist_ok=True)

        videos = sorted(
            p for p in group_video_dir.iterdir()
            if p.suffix.lower() in VIDEO_EXTENSIONS
        )

        if not videos:
            logger.warning("No videos found in %s", group_video_dir)
            continue

        logger.info("Processing group '%s': %d videos", group, len(videos))
        saved_paths: List[Path] = []

        for video_path in videos:
            npy_path = group_flow_dir / f"{video_path.stem}.npy"
            fps_path = group_flow_dir / f"{video_path.stem}_fps.npy"

            if npy_path.exists():
                logger.info("  cached: %s", npy_path.name)
                saved_paths.append(npy_path)
                continue

            try:
                flow_fields, fps = extract_flow(video_path, device=device)
                np.save(npy_path, flow_fields)
                np.save(fps_path, np.array(fps))
                saved_paths.append(npy_path)
                logger.info("  saved: %s (%s)", npy_path.name, flow_fields.shape)
            except Exception:
                logger.exception("  FAILED: %s", video_path.name)

        result[group] = saved_paths

    return result
