#!/usr/bin/env python3
"""
H100-optimized RAFT optical flow extraction.

Designed for high-VRAM GPUs (80-96 GB). Key optimizations over the
single-pair RTX pipeline:

  1. Batched inference — process B frame pairs per forward pass
  2. Threaded frame decoding — CPU reads ahead while GPU computes
  3. torch.compile — Hopper-native kernel fusion (~1.5x)
  4. BF16 — native H100 precision

Expected throughput: ~40-60 pairs/sec on H100 SXM (vs 3/sec on RTX 5080).
537K total frame pairs → ~2.5-4 hours instead of 50.

Usage:
    # Full extraction
    python extract_h100.py --video-dir data/videos --output-dir output

    # Resume after interruption (skips existing .npy files)
    python extract_h100.py --video-dir data/videos --output-dir output

    # Custom batch size (higher = faster but more VRAM)
    python extract_h100.py --video-dir data/videos --output-dir output --batch-size 24

    # Specific groups only
    python extract_h100.py --video-dir data/videos --output-dir output --groups seizure normal
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from threading import Thread
from typing import List, Optional

import cv2
import numpy as np
import torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("h100_extract")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


# ---------------------------------------------------------------------------
# Threaded frame reader — decodes on CPU while GPU runs RAFT
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Batched preprocessing
# ---------------------------------------------------------------------------

def preprocess_batch(
    frames: List[np.ndarray], device: torch.device,
) -> torch.Tensor:
    """Convert a list of BGR frames to a batched RAFT input tensor.

    Returns [B, 3, H_padded, W_padded] float32 on device.
    All frames must have the same spatial dimensions.
    """
    tensors = []
    for bgr in frames:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensors.append(t)

    batch = torch.stack(tensors)  # [B, 3, H, W]
    _, _, h, w = batch.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h > 0 or pad_w > 0:
        batch = torch.nn.functional.pad(batch, (0, pad_w, 0, pad_h))
    return batch.to(device)


# ---------------------------------------------------------------------------
# Core extraction — batched + memmap
# ---------------------------------------------------------------------------

def extract_flow_batched(
    video_path: Path,
    output_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 16,
) -> tuple[Path, float]:
    """Extract optical flow with batched RAFT inference.

    Reads frames in a background thread, batches B consecutive pairs,
    runs RAFT once per batch, writes results to a memory-mapped .npy.

    Args:
        video_path: Input video.
        output_path: Where to write the [N-1, H, W, 2] float32 .npy.
        model: Pre-loaded (and optionally compiled) RAFT model.
        device: CUDA device.
        batch_size: Number of frame pairs per forward pass.

    Returns:
        (output_path, fps)
    """
    reader = FrameReader(video_path, buffer_size=batch_size * 3).start()
    fps = reader.fps
    n_total = reader.n_frames
    n_pairs = max(n_total - 1, 1)

    logger.info(
        "Video: %s — %d frames, %.1f fps, batch=%d",
        video_path.name, n_total, fps, batch_size,
    )

    # Read first frame
    prev_frame = reader.next_frame()
    if prev_frame is None:
        raise ValueError(f"Cannot read first frame: {video_path}")

    orig_h, orig_w = prev_frame.shape[:2]

    # Pre-allocate memmap on disk
    memmap = np.lib.format.open_memmap(
        str(output_path), mode="w+", dtype=np.float32,
        shape=(n_pairs, orig_h, orig_w, 2),
    )

    frame_idx = 0
    t_start = time.perf_counter()

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        while True:
            # Collect a batch of frame pairs
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
            t1 = preprocess_batch(prev_frames, device)  # [B, 3, H, W]
            t2 = preprocess_batch(curr_frames, device)

            flow_preds = model(t1, t2)
            flow_batch = flow_preds[-1]  # [B, 2, H_pad, W_pad]
            flow_batch = flow_batch[:, :, :orig_h, :orig_w]  # [B, 2, H, W]

            # Write directly to memmap — no RAM accumulation
            flow_np = flow_batch.cpu().numpy().transpose(0, 2, 3, 1)  # [B, H, W, 2]
            memmap[frame_idx : frame_idx + B] = flow_np

            frame_idx += B

            if frame_idx % (batch_size * 10) == 0 or frame_idx >= n_pairs - batch_size:
                elapsed = time.perf_counter() - t_start
                rate = frame_idx / elapsed
                eta = (n_pairs - frame_idx) / rate if rate > 0 else 0
                logger.info(
                    "  %d/%d pairs (%.1f pairs/sec, ETA %.0fs)",
                    frame_idx, n_pairs, rate, eta,
                )

    # Handle truncation if fewer frames than metadata claimed
    if frame_idx == 0:
        del memmap
        output_path.unlink(missing_ok=True)
        raise ValueError(f"Video has <2 readable frames: {video_path}")

    if frame_idx < n_pairs:
        logger.info("  truncating %d → %d pairs", n_pairs, frame_idx)
        del memmap
        src = np.lib.format.open_memmap(str(output_path), mode="r")
        trunc_path = str(output_path) + ".trunc"
        dst = np.lib.format.open_memmap(
            trunc_path, mode="w+", dtype=np.float32,
            shape=(frame_idx, orig_h, orig_w, 2),
        )
        chunk = 64
        for i in range(0, frame_idx, chunk):
            dst[i : i + chunk] = src[i : i + chunk]
        del src, dst
        Path(trunc_path).replace(output_path)
    else:
        del memmap

    elapsed = time.perf_counter() - t_start
    logger.info(
        "  done: %d pairs in %.1fs (%.1f pairs/sec)",
        frame_idx, elapsed, frame_idx / elapsed,
    )
    return output_path, fps


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="H100-optimized RAFT optical flow extraction",
    )
    parser.add_argument(
        "--video-dir", type=Path, required=True,
        help="Root directory with {group}/ subdirs containing videos",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"),
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Frame pairs per RAFT forward pass (default: 16, max ~24 for 80GB)",
    )
    parser.add_argument(
        "--groups", nargs="+", default=None,
        help="Process only these groups (default: all subdirs)",
    )
    parser.add_argument(
        "--no-compile", action="store_true",
        help="Skip torch.compile (useful for debugging)",
    )
    parser.add_argument(
        "--compact", action="store_true",
        help="Save compact representations (128x128 flow + time series) and delete full-res flows",
    )
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    device = torch.device("cuda")
    logger.info("Device: %s", torch.cuda.get_device_name(device))
    logger.info(
        "VRAM: %.1f GB total",
        torch.cuda.get_device_properties(device).total_mem / 1e9,
    )

    # Load model
    logger.info("Loading RAFT-Large...")
    weights = Raft_Large_Weights.C_T_SKHT_V2
    model = raft_large(weights=weights).to(device).eval()

    if not args.no_compile:
        logger.info("Compiling model with torch.compile (max-autotune)...")
        model = torch.compile(model, mode="max-autotune")
        # Warmup — first call triggers compilation
        dummy = torch.randn(1, 3, 520, 960, device=device)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            _ = model(dummy, dummy)
        logger.info("Compilation done.")

    # Discover videos
    video_dir = args.video_dir
    flow_dir = args.output_dir / "flows"
    flow_dir.mkdir(parents=True, exist_ok=True)

    compact_dir: Optional[Path] = None
    if args.compact:
        compact_dir = args.output_dir / "compact"
        compact_dir.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(Path(__file__).parent))
        from compute_summaries import save_compact_representation

    if args.groups is None:
        groups = sorted(
            d.name for d in video_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
    else:
        groups = args.groups

    total_videos = 0
    total_done = 0
    total_skipped = 0
    grand_start = time.perf_counter()

    for group in groups:
        group_video_dir = video_dir / group
        if not group_video_dir.exists():
            logger.warning("Group dir not found: %s", group_video_dir)
            continue

        group_flow_dir = flow_dir / group
        group_flow_dir.mkdir(parents=True, exist_ok=True)

        videos = sorted(
            p for p in group_video_dir.iterdir()
            if p.suffix.lower() in VIDEO_EXTENSIONS
        )

        if not videos:
            logger.warning("No videos in %s", group_video_dir)
            continue

        logger.info("=" * 60)
        logger.info("Group '%s': %d videos", group, len(videos))
        logger.info("=" * 60)

        for video_path in videos:
            total_videos += 1
            npy_path = group_flow_dir / f"{video_path.stem}.npy"
            fps_path = group_flow_dir / f"{video_path.stem}_fps.npy"

            # Skip if already processed (full flow or compact)
            compact_exists = (
                compact_dir is not None
                and (compact_dir / group / f"{video_path.stem}_mag_ts.npy").exists()
            )
            if npy_path.exists() or compact_exists:
                logger.info("  cached: %s", video_path.stem)
                total_skipped += 1
                continue

            tmp_path = npy_path.with_suffix(".npy.tmp")
            try:
                _, fps = extract_flow_batched(
                    video_path, tmp_path, model, device,
                    batch_size=args.batch_size,
                )
                tmp_path.rename(npy_path)
                np.save(fps_path, np.array(fps))
                total_done += 1

                if compact_dir is not None:
                    group_compact = compact_dir / group
                    save_compact_representation(
                        npy_path, group_compact, video_path.stem,
                    )
                    np.save(group_compact / f"{video_path.stem}_fps.npy",
                            np.array(fps))
                    npy_path.unlink()
                    fps_path.unlink(missing_ok=True)
                    logger.info("  deleted full flow: %s", npy_path.name)
            except Exception:
                logger.exception("  FAILED: %s", video_path.name)
            finally:
                tmp_path.unlink(missing_ok=True)
                torch.cuda.empty_cache()

    elapsed = time.perf_counter() - grand_start
    logger.info("=" * 60)
    logger.info(
        "COMPLETE: %d extracted, %d cached, %d total in %.0f min",
        total_done, total_skipped, total_videos, elapsed / 60,
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
