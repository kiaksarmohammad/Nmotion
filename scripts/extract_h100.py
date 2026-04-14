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

TARGET_H, TARGET_W = 520, 960  # Divisible by 8 — no padding, no recompilation


def preprocess_batch(
    frames: List[np.ndarray], device: torch.device,
) -> torch.Tensor:
    """Convert a list of BGR frames to a batched RAFT input tensor.

    Resizes all frames to TARGET_H x TARGET_W so torch.compile only traces once.
    Returns [B, 3, TARGET_H, TARGET_W] float32 on device.
    """
    tensors = []
    for bgr in frames:
        bgr = cv2.resize(bgr, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensors.append(t)

    return torch.stack(tensors).to(device)  # [B, 3, TARGET_H, TARGET_W]


# ---------------------------------------------------------------------------
# Core extraction — batched + memmap
# ---------------------------------------------------------------------------

def _run_batched_inference(
    video_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    on_batch,
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
    """Extract optical flow to disk via memory-mapped .npy.

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
        src = np.lib.format.open_memmap(str(output_path), mode="r")
        trunc_path = str(output_path) + ".trunc"
        dst = np.lib.format.open_memmap(
            trunc_path, mode="w+", dtype=np.float32,
            shape=(frame_idx, proc_h, proc_w, 2),
        )
        chunk = 64
        for i in range(0, frame_idx, chunk):
            dst[i : i + chunk] = src[i : i + chunk]
        del src, dst
        Path(trunc_path).replace(output_path)
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

    proc_h, proc_w = TARGET_H, TARGET_W
    scale_x = target_size / proc_w
    scale_y = target_size / proc_h

    # Accumulators for compact representations
    mag_ts_chunks: List[np.ndarray] = []
    spatial_chunks: List[np.ndarray] = []
    flow128_chunks: List[np.ndarray] = []

    def on_batch(flow_np: np.ndarray, idx: int) -> None:
        """Compute compact summaries from this batch's flow output."""
        B = flow_np.shape[0]  # [B, TARGET_H, TARGET_W, 2]

        # --- Magnitude time series [B, 6] ---
        stats = np.empty((B, 6), dtype=np.float32)
        for i in range(B):
            mag = np.sqrt(flow_np[i, :, :, 0] ** 2 + flow_np[i, :, :, 1] ** 2)
            flat = mag.ravel()
            stats[i, 0] = flat.mean()
            stats[i, 1] = flat.max()
            stats[i, 2] = flat.std()
            stats[i, 3] = np.median(flat)
            stats[i, 4] = np.percentile(flat, 5)
            stats[i, 5] = np.percentile(flat, 95)
        mag_ts_chunks.append(stats)

        # --- Spatial summary [B, 12] ---
        mid_h, mid_w = proc_h // 2, proc_w // 2
        summary = np.empty((B, 12), dtype=np.float32)
        for i in range(B):
            u, v = flow_np[i, :, :, 0], flow_np[i, :, :, 1]
            mag = np.sqrt(u ** 2 + v ** 2)

            summary[i, 0] = mag[:mid_h, :mid_w].mean()
            summary[i, 1] = mag[:mid_h, mid_w:].mean()
            summary[i, 2] = mag[mid_h:, :mid_w].mean()
            summary[i, 3] = mag[mid_h:, mid_w:].mean()

            left_energy = (mag[:, :mid_w] ** 2).sum()
            right_energy = (mag[:, mid_w:] ** 2).sum()
            total = left_energy + right_energy
            summary[i, 4] = left_energy / total if total > 0 else 0.5

            dvdx = np.gradient(v, axis=1)
            dudy = np.gradient(u, axis=0)
            summary[i, 5] = (dvdx - dudy).mean()

            dudx = np.gradient(u, axis=1)
            dvdy = np.gradient(v, axis=0)
            summary[i, 6] = (dudx + dvdy).mean()

            mean_u, mean_v = u.mean(), v.mean()
            mean_mag = np.sqrt(mean_u ** 2 + mean_v ** 2)
            if mean_mag > 1e-6:
                dot = (u * mean_u + v * mean_v) / (mag * mean_mag + 1e-8)
                summary[i, 7] = dot.mean()
            else:
                summary[i, 7] = 0.0

            summary[i, 8] = u[:mid_h, :].mean() - u[mid_h:, :].mean()
            summary[i, 9] = v[:mid_h, :].mean() - v[mid_h:, :].mean()
            summary[i, 10] = u[:, :mid_w].mean() - u[:, mid_w:].mean()
            summary[i, 11] = v[:, :mid_w].mean() - v[:, mid_w:].mean()
        spatial_chunks.append(summary)

        # --- Downscaled flow [B, 128, 128, 2] float16 ---
        small = np.empty((B, target_size, target_size, 2), dtype=np.float16)
        for i in range(B):
            resized = cv2.resize(flow_np[i], (target_size, target_size),
                                 interpolation=cv2.INTER_AREA)
            resized[:, :, 0] *= scale_x
            resized[:, :, 1] *= scale_y
            small[i] = resized.astype(np.float16)
        flow128_chunks.append(small)

    frame_idx, fps, _ = _run_batched_inference(
        video_path, model, device, batch_size, on_batch,
    )

    # Concatenate and save
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
        "--batch-size", type=int, default=24,
        help="Frame pairs per RAFT forward pass (default: 24, max due to cuDNN grid_sample limit)",
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
        "--num-workers", type=int, default=1,
        help="Number of parallel processes sharing the GPU (each loads its own model)",
    )
    parser.add_argument(
        "--worker-id", type=int, default=0,
        help="This worker's index (0..num-workers-1); each processes every N-th video",
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
        torch.cuda.get_device_properties(device).total_memory / 1e9,
    )

    # Load model
    logger.info("Loading RAFT-Large...")
    weights = Raft_Large_Weights.C_T_SKHT_V2
    model = raft_large(weights=weights).to(device).eval()

    if not args.no_compile:
        logger.info("Compiling model with torch.compile (max-autotune)...")
        model = torch.compile(model, mode="max-autotune")
        # Warmup at exact batch_size + target resolution — compile once, run forever
        logger.info("Warmup: batch=%d, resolution=%dx%d", args.batch_size, TARGET_H, TARGET_W)
        dummy = torch.randn(args.batch_size, 3, TARGET_H, TARGET_W, device=device)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            _ = model(dummy, dummy)
        del dummy
        torch.cuda.empty_cache()
        logger.info("Compilation done.")

    # Discover videos
    video_dir = args.video_dir
    flow_dir = args.output_dir / "flows"
    flow_dir.mkdir(parents=True, exist_ok=True)

    compact_dir: Optional[Path] = None
    if args.compact:
        compact_dir = args.output_dir / "compact"
        compact_dir.mkdir(parents=True, exist_ok=True)

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

        all_videos = sorted(
            p for p in group_video_dir.iterdir()
            if p.suffix.lower() in VIDEO_EXTENSIONS
        )

        # Each worker takes every N-th video
        videos = [v for i, v in enumerate(all_videos)
                  if i % args.num_workers == args.worker_id]

        if not videos:
            logger.warning("No videos for worker %d in %s", args.worker_id, group_video_dir)
            continue

        logger.info("=" * 60)
        logger.info("Group '%s': %d/%d videos (worker %d/%d)",
                    group, len(videos), len(all_videos),
                    args.worker_id, args.num_workers)
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

            try:
                if compact_dir is not None:
                    # Streaming: compute compact representations during
                    # inference — never writes full flow to disk
                    group_compact = compact_dir / group
                    extract_flow_compact_streaming(
                        video_path, group_compact, video_path.stem,
                        model, device, batch_size=args.batch_size,
                    )
                else:
                    # Legacy: write full flow to memmap
                    tmp_path = npy_path.with_suffix(".npy.tmp")
                    _, fps = extract_flow_batched(
                        video_path, tmp_path, model, device,
                        batch_size=args.batch_size,
                    )
                    tmp_path.rename(npy_path)
                    np.save(fps_path, np.array(fps))
                total_done += 1
            except Exception:
                logger.exception("  FAILED: %s", video_path.name)
            finally:
                # Clean up any leftover tmp from non-compact mode
                tmp_path_cleanup = npy_path.with_suffix(".npy.tmp")
                tmp_path_cleanup.unlink(missing_ok=True)
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
