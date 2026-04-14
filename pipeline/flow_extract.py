"""Dense optical flow extraction using torchvision RAFT.

Two extraction paths, chosen per-GPU:

  * Single-pair (RTX-class) — `extract_flow`, `extract_all_flows`
      Processes one frame pair at a time at native resolution.
      Pads inputs to multiples of 8. Peak RAM ≈ 2 frames + 1 flow field.

  * Batched / compiled (H100-class) — `extract_flow_batched`,
    `extract_flow_compact_streaming`
      Processes B frame pairs per forward pass at a fixed 520×960
      resolution so `torch.compile` only traces once. The streaming
      variant computes compact summaries (`pipeline.compact`) during
      inference and never writes the full-res flow to disk.

RC-1: swap in SEA-RAFT or FlowSeek if RAFT-Large quality is insufficient
on neonatal video.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from threading import Thread
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F  # noqa: F401 — exported for callers
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

from pipeline.compact import (
    compute_magnitude_timeseries,
    compute_spatial_summary,
    downscale_flow,
)

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

# Fixed processing resolution for the batched path — divisible by 8 so
# RAFT needs no padding, and torch.compile only traces once.
TARGET_H, TARGET_W = 520, 960


# ═══════════════════════════════════════════════════════════════════════════
# Single-pair (RTX) path
# ═══════════════════════════════════════════════════════════════════════════

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
        _truncate_memmap(output_path, frame_idx, orig_h, orig_w)
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


# ═══════════════════════════════════════════════════════════════════════════
# Batched (H100) path — fixed resolution, torch.compile-friendly
# ═══════════════════════════════════════════════════════════════════════════

class FrameReader:
    """Reads frames from a video in a background thread.

    Maintains a buffer of decoded BGR frames so the GPU never waits
    on cv2.VideoCapture.read().
    """

    def __init__(self, video_path: Path, buffer_size: int = 64):
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.buffer: deque[Optional[np.ndarray]] = deque(maxlen=buffer_size)
        self._done = False
        self._thread = Thread(target=self._read_loop, daemon=True)

    def start(self) -> "FrameReader":
        self._thread.start()
        return self

    def _read_loop(self) -> None:
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.buffer.append(None)  # sentinel
                break
            # Spin-wait if buffer is full (rare — GPU is usually slower)
            while len(self.buffer) == self.buffer.maxlen:
                time.sleep(0.001)
            self.buffer.append(frame)
        self.cap.release()
        self._done = True

    def next_frame(self) -> Optional[np.ndarray]:
        """Return next frame, or None at end of video."""
        while len(self.buffer) == 0 and not self._done:
            time.sleep(0.001)
        if len(self.buffer) == 0:
            return None
        return self.buffer.popleft()


def preprocess_batch(
    frames: List[np.ndarray], device: torch.device,
) -> torch.Tensor:
    """Convert a list of BGR frames to a batched RAFT input tensor.

    Resizes all frames to TARGET_H × TARGET_W so torch.compile only traces once.
    Returns [B, 3, TARGET_H, TARGET_W] float32 on device.
    """
    tensors = []
    for bgr in frames:
        bgr = cv2.resize(bgr, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensors.append(t)

    return torch.stack(tensors).to(device)  # [B, 3, TARGET_H, TARGET_W]


def _run_batched_inference(
    video_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    on_batch: Callable[[np.ndarray, int], None],
) -> tuple[int, float, float]:
    """Shared inference loop — calls on_batch(flow_np, frame_idx) per batch.

    Returns (frame_idx, fps, elapsed).
    """
    reader = FrameReader(video_path, buffer_size=batch_size * 3).start()
    fps = reader.fps
    n_total = reader.n_frames
    n_pairs = max(n_total - 1, 1)

    logger.info(
        "Video: %s — %d frames, %.1f fps, batch=%d",
        video_path.name, n_total, fps, batch_size,
    )

    prev_frame = reader.next_frame()
    if prev_frame is None:
        raise ValueError(f"Cannot read first frame: {video_path}")

    frame_idx = 0
    t_start = time.perf_counter()

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        while True:
            prev_frames: List[np.ndarray] = []
            curr_frames: List[np.ndarray] = []

            for _ in range(batch_size):
                curr_frame = reader.next_frame()
                if curr_frame is None:
                    break
                prev_frames.append(prev_frame)
                curr_frames.append(curr_frame)
                prev_frame = curr_frame

            if not prev_frames:
                break

            B = len(prev_frames)

            if B < batch_size:
                prev_frames.extend([prev_frames[-1]] * (batch_size - B))
                curr_frames.extend([curr_frames[-1]] * (batch_size - B))

            t1 = preprocess_batch(prev_frames, device)
            t2 = preprocess_batch(curr_frames, device)

            flow_preds = model(t1, t2)
            flow_batch = flow_preds[-1]  # [batch_size, 2, TARGET_H, TARGET_W]

            flow_np = flow_batch[:B].cpu().numpy().transpose(0, 2, 3, 1)
            on_batch(flow_np, frame_idx)

            frame_idx += B

            if frame_idx % (batch_size * 10) == 0 or frame_idx >= n_pairs - batch_size:
                elapsed = time.perf_counter() - t_start
                rate = frame_idx / elapsed
                eta = (n_pairs - frame_idx) / rate if rate > 0 else 0
                logger.info(
                    "  %d/%d pairs (%.1f pairs/sec, ETA %.0fs)",
                    frame_idx, n_pairs, rate, eta,
                )

    elapsed = time.perf_counter() - t_start
    if frame_idx == 0:
        raise ValueError(f"Video has <2 readable frames: {video_path}")

    logger.info(
        "  done: %d pairs in %.1fs (%.1f pairs/sec)",
        frame_idx, elapsed, frame_idx / elapsed,
    )
    return frame_idx, fps, elapsed


def extract_flow_batched(
    video_path: Path,
    output_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 16,
) -> tuple[Path, float]:
    """Extract optical flow to disk via memory-mapped .npy, batched.

    Returns (output_path, fps).
    """
    reader_peek = FrameReader(video_path, buffer_size=2).start()
    n_pairs = max(reader_peek.n_frames - 1, 1)
    # Release — the real reader is created inside _run_batched_inference
    reader_peek.cap.release()

    proc_h, proc_w = TARGET_H, TARGET_W
    memmap = np.lib.format.open_memmap(
        str(output_path), mode="w+", dtype=np.float32,
        shape=(n_pairs, proc_h, proc_w, 2),
    )

    def on_batch(flow_np: np.ndarray, idx: int) -> None:
        memmap[idx : idx + flow_np.shape[0]] = flow_np

    frame_idx, fps, _ = _run_batched_inference(
        video_path, model, device, batch_size, on_batch,
    )

    if frame_idx < n_pairs:
        logger.info("  truncating %d → %d pairs", n_pairs, frame_idx)
        del memmap
        _truncate_memmap(output_path, frame_idx, proc_h, proc_w)
    else:
        del memmap

    return output_path, fps


def extract_flow_compact_streaming(
    video_path: Path,
    compact_dir: Path,
    stem: str,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 16,
    target_size: int = 128,
) -> float:
    """Extract optical flow and compute compact representations in one pass.

    Never writes the full-resolution flow to disk — computes mag_ts, spatial
    summary, and downscaled flow batch-by-batch during inference. Peak disk
    usage is ~50 MB per video instead of potentially 30+ GB.

    Returns fps.
    """
    compact_dir.mkdir(parents=True, exist_ok=True)

    mag_ts_chunks: List[np.ndarray] = []
    spatial_chunks: List[np.ndarray] = []
    flow128_chunks: List[np.ndarray] = []

    def on_batch(flow_np: np.ndarray, idx: int) -> None:
        """Compute compact summaries from this batch's flow output."""
        mag_ts_chunks.append(compute_magnitude_timeseries(flow_np))
        spatial_chunks.append(compute_spatial_summary(flow_np))
        flow128_chunks.append(downscale_flow(flow_np, target_size))

    frame_idx, fps, _ = _run_batched_inference(
        video_path, model, device, batch_size, on_batch,
    )

    mag_ts = np.concatenate(mag_ts_chunks)
    spatial = np.concatenate(spatial_chunks)
    flow128 = np.concatenate(flow128_chunks)

    np.save(compact_dir / f"{stem}_mag_ts.npy", mag_ts)
    np.save(compact_dir / f"{stem}_spatial.npy", spatial)
    np.save(compact_dir / f"{stem}_flow128.npy", flow128)
    np.save(compact_dir / f"{stem}_fps.npy", np.array(fps))

    size_mb = (mag_ts.nbytes + spatial.nbytes + flow128.nbytes) / 1e6
    logger.info("  compact saved: %.1f MB (%d frames)", size_mb, frame_idx)

    return fps


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _truncate_memmap(
    output_path: Path, frame_idx: int, h: int, w: int,
) -> None:
    """Truncate a memmapped flow file to frame_idx pairs, in-place via rename."""
    src = np.lib.format.open_memmap(str(output_path), mode="r")
    trunc_path = str(output_path) + ".trunc"
    dst = np.lib.format.open_memmap(
        trunc_path, mode="w+", dtype=np.float32,
        shape=(frame_idx, h, w, 2),
    )
    chunk = 64
    for i in range(0, frame_idx, chunk):
        dst[i : i + chunk] = src[i : i + chunk]
    del src, dst
    Path(trunc_path).replace(output_path)
