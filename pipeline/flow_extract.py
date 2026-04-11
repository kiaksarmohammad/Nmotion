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


def _preprocess_frame(bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert a single BGR frame to a RAFT input tensor.

    RAFT expects float32 tensors in [0, 1] range, shape [1, 3, H, W].
    Pads to dimensions divisible by 8 (RAFT requirement).
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    _, h, w = t.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h > 0 or pad_w > 0:
        t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="constant")
    return t.unsqueeze(0).to(device)


def _load_raft(device: torch.device) -> torch.nn.Module:
    """Load RAFT-Large once; reuse across videos."""
    if device.type == "cuda":
        import os
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        )
    weights = Raft_Large_Weights.C_T_SKHT_V2
    return raft_large(weights=weights).to(device).eval()


def extract_flow(
    video_path: Path,
    output_path: Path,
    device: str = "cuda",
    model: Optional[torch.nn.Module] = None,
) -> tuple[Path, float]:
    """Extract dense optical flow from a video using RAFT-Large.

    Streams frames pair-by-pair and writes each flow field directly to a
    memory-mapped .npy file on disk.  Peak RAM ≈ 2 × one frame (prev + curr)
    + one flow field — independent of video length.

    Args:
        video_path: Path to video file.
        output_path: Where to save the [N-1, H, W, 2] float32 .npy file.
        device: "cuda" or "cpu".
        model: Pre-loaded RAFT model (avoids reloading per video).

    Returns:
        output_path: Path to the saved .npy file.
        fps: video frame rate.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    if model is None:
        logger.info("Loading RAFT-Large on %s", dev)
        model = _load_raft(dev)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("Video: %s — %d frames, %.1f fps", video_path.name, n_total, fps)

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError(f"Cannot read first frame: {video_path}")

    orig_h, orig_w = prev_frame.shape[:2]
    n_pairs = max(n_total - 1, 1)

    # Pre-allocate a memory-mapped file so flow is written directly to disk
    # instead of accumulating in RAM (1680 frames × 1920×1080×2×4B = 27 GB)
    memmap = np.lib.format.open_memmap(
        str(output_path), mode="w+", dtype=np.float32,
        shape=(n_pairs, orig_h, orig_w, 2),
    )

    frame_idx = 0
    with torch.no_grad(), torch.amp.autocast(device_type=dev.type, dtype=torch.float16):
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            t1 = _preprocess_frame(prev_frame, dev)
            t2 = _preprocess_frame(curr_frame, dev)

            flow_predictions = model(t1, t2)
            flow = flow_predictions[-1]  # [1, 2, H, W]
            flow = flow[0, :, :orig_h, :orig_w]  # [2, H, W]

            memmap[frame_idx] = flow.cpu().numpy().transpose(1, 2, 0)

            prev_frame = curr_frame
            frame_idx += 1

            if frame_idx % 100 == 0:
                logger.info("  processed %d/%d frame pairs", frame_idx, n_pairs)

    cap.release()

    if frame_idx == 0:
        del memmap
        output_path.unlink(missing_ok=True)
        raise ValueError(f"Video has <2 readable frames: {video_path}")

    # Truncate if video had fewer readable frames than CAP_PROP_FRAME_COUNT
    if frame_idx < n_pairs:
        logger.info("  truncating %d → %d pairs (video ended early)", n_pairs, frame_idx)
        del memmap
        src = np.lib.format.open_memmap(str(output_path), mode="r")
        trunc_path = str(output_path) + ".trunc"
        dst = np.lib.format.open_memmap(
            trunc_path, mode="w+", dtype=np.float32,
            shape=(frame_idx, orig_h, orig_w, 2),
        )
        # Copy in chunks to stay RAM-friendly
        chunk = 64
        for i in range(0, frame_idx, chunk):
            dst[i : i + chunk] = src[i : i + chunk]
        del src, dst
        Path(trunc_path).replace(output_path)
    else:
        del memmap

    logger.info("Flow extraction complete: (%d, %d, %d, 2)", frame_idx, orig_h, orig_w)
    return output_path, fps


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

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info("Loading RAFT-Large on %s", dev)
    model = _load_raft(dev)

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

            tmp_path = npy_path.with_suffix(".npy.tmp")
            try:
                _, fps = extract_flow(
                    video_path, tmp_path, device=device, model=model,
                )
                tmp_path.rename(npy_path)
                np.save(fps_path, np.array(fps))
                saved_paths.append(npy_path)
                logger.info("  saved: %s", npy_path.name)
            except Exception:
                logger.exception("  FAILED: %s", video_path.name)
            finally:
                tmp_path.unlink(missing_ok=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        result[group] = saved_paths

    return result
