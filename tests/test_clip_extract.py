"""Tests for temporal clip extraction from flow fields."""

import numpy as np
import pytest

from pipeline.clip_extract import extract_clips, extract_all_clips


class TestExtractClips:
    """Test sliding-window clip extraction on synthetic flow arrays."""

    def test_basic_extraction(self):
        # 100 frames, 4x4 spatial, 2 channels
        flow = np.random.randn(100, 4, 4, 2).astype(np.float32)
        clips = extract_clips(flow, window_frames=30, overlap=0.5)
        # stride = 30 * 0.5 = 15, positions: 0,15,30,45,60 → 5 clips (last start at 70)
        assert len(clips) == 5
        for clip in clips:
            assert clip.shape == (30, 4, 4, 2)

    def test_no_overlap(self):
        flow = np.random.randn(90, 4, 4, 2).astype(np.float32)
        clips = extract_clips(flow, window_frames=30, overlap=0.0)
        assert len(clips) == 3

    def test_high_overlap(self):
        flow = np.random.randn(100, 4, 4, 2).astype(np.float32)
        clips = extract_clips(flow, window_frames=30, overlap=0.9)
        # stride = 3, positions: 0,3,6,...,69 → 24 clips
        assert len(clips) == 24

    def test_flow_shorter_than_window(self):
        flow = np.random.randn(10, 4, 4, 2).astype(np.float32)
        clips = extract_clips(flow, window_frames=30, overlap=0.5)
        # Falls back to single clip (entire flow, zero-padded or returned as-is)
        assert len(clips) == 1
        assert clips[0].shape[0] == 10

    def test_clip_content_is_correct_slice(self):
        flow = np.arange(60).reshape(60, 1, 1, 1).astype(np.float32)
        flow = np.broadcast_to(flow, (60, 1, 1, 2)).copy()
        clips = extract_clips(flow, window_frames=20, overlap=0.0)
        assert len(clips) == 3
        # First clip: frames 0-19
        np.testing.assert_array_equal(clips[0][:, 0, 0, 0], np.arange(20))
        # Second clip: frames 20-39
        np.testing.assert_array_equal(clips[1][:, 0, 0, 0], np.arange(20, 40))


from pipeline.features import extract_clip_features


class TestClipFeatures:
    def test_returns_dataframe_with_video_id(self):
        rng = np.random.default_rng(42)
        clips = [rng.standard_normal((50, 4, 4, 2)).astype(np.float32) for _ in range(3)]
        labels = ["seizure", "seizure", "normal"]
        video_ids = ["vid1", "vid1", "vid2"]
        fps_values = [30.0, 30.0, 30.0]

        df = extract_clip_features(clips, labels, video_ids, fps_values)
        assert len(df) == 3
        assert "video_id" in df.columns
        assert "group" in df.columns
        assert "sample_entropy" in df.columns

    def test_short_clip_no_crash(self):
        """Clips shorter than ideal should still produce features (with NaN MSE)."""
        clip = np.random.randn(20, 4, 4, 2).astype(np.float32)
        df = extract_clip_features([clip], ["normal"], ["v1"], [30.0])
        assert len(df) == 1
