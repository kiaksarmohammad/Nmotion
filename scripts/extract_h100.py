#!/usr/bin/env python3
"""H100-optimized RAFT optical flow extraction (CLI wrapper).

All heavy lifting lives in `pipeline.flow_extract`; this file just handles
argparse, model load + torch.compile warmup, and the per-video dispatch.

Expected throughput on H100 SXM: ~40-60 pairs/sec (vs 3/sec on RTX 5080).

Usage:
    # Full extraction (full-res memmap flows)
    python scripts/extract_h100.py --video-dir data/videos --output-dir output

    # Streaming compact mode (recommended for large datasets)
    python scripts/extract_h100.py --video-dir data/videos --output-dir output --compact

    # Custom batch size (higher = faster but more VRAM)
    python scripts/extract_h100.py --video-dir data/videos --output-dir output --batch-size 24

    # Specific groups only
    python scripts/extract_h100.py --video-dir data/videos --output-dir output --groups seizure normal

    # Multi-worker (each worker processes every N-th video, loads its own model)
    python scripts/extract_h100.py --video-dir data/videos --output-dir output \
        --num-workers 2 --worker-id 0
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

# Allow `python scripts/extract_h100.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.flow_extract import (
    TARGET_H,
    TARGET_W,
    VIDEO_EXTENSIONS,
    extract_flow_batched,
    extract_flow_compact_streaming,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("h100_extract")


def parse_args() -> argparse.Namespace:
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
        help="Frame pairs per RAFT forward pass (default: 24)",
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
        help="Number of parallel processes sharing the GPU",
    )
    parser.add_argument(
        "--worker-id", type=int, default=0,
        help="This worker's index (0..num-workers-1)",
    )
    parser.add_argument(
        "--compact", action="store_true",
        help="Save compact representations (no full-res flow on disk)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    device = torch.device("cuda")
    logger.info("Device: %s", torch.cuda.get_device_name(device))
    logger.info(
        "VRAM: %.1f GB total",
        torch.cuda.get_device_properties(device).total_memory / 1e9,
    )

    logger.info("Loading RAFT-Large...")
    weights = Raft_Large_Weights.C_T_SKHT_V2
    model = raft_large(weights=weights).to(device).eval()

    if not args.no_compile:
        logger.info("Compiling model with torch.compile (max-autotune)...")
        model = torch.compile(model, mode="max-autotune")
        logger.info(
            "Warmup: batch=%d, resolution=%dx%d",
            args.batch_size, TARGET_H, TARGET_W,
        )
        dummy = torch.randn(
            args.batch_size, 3, TARGET_H, TARGET_W, device=device,
        )
        with torch.no_grad(), torch.amp.autocast(
            device_type="cuda", dtype=torch.bfloat16,
        ):
            _ = model(dummy, dummy)
        del dummy
        torch.cuda.empty_cache()
        logger.info("Compilation done.")

    video_dir = args.video_dir
    flow_dir = args.output_dir / "flows"
    flow_dir.mkdir(parents=True, exist_ok=True)

    compact_dir: Path | None = None
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

        videos = [
            v for i, v in enumerate(all_videos)
            if i % args.num_workers == args.worker_id
        ]

        if not videos:
            logger.warning(
                "No videos for worker %d in %s",
                args.worker_id, group_video_dir,
            )
            continue

        logger.info("=" * 60)
        logger.info(
            "Group '%s': %d/%d videos (worker %d/%d)",
            group, len(videos), len(all_videos),
            args.worker_id, args.num_workers,
        )
        logger.info("=" * 60)

        for video_path in videos:
            total_videos += 1
            npy_path = group_flow_dir / f"{video_path.stem}.npy"
            fps_path = group_flow_dir / f"{video_path.stem}_fps.npy"

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
                    group_compact = compact_dir / group
                    extract_flow_compact_streaming(
                        video_path, group_compact, video_path.stem,
                        model, device, batch_size=args.batch_size,
                    )
                else:
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
                tmp_cleanup = npy_path.with_suffix(".npy.tmp")
                tmp_cleanup.unlink(missing_ok=True)
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
