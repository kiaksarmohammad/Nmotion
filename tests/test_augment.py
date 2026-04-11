"""Tests for flow field augmentation functions."""

import numpy as np
import pytest

from pipeline.augment import (
    augment_gaussian_noise,
    augment_horizontal_flip,
    augment_temporal_reverse,
    augment_magnitude_scale,
    apply_augmentations,
)


@pytest.fixture
def sample_clip():
    """A small synthetic flow clip [30, 8, 8, 2]."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((30, 8, 8, 2)).astype(np.float32)


class TestGaussianNoise:
    def test_shape_preserved(self, sample_clip):
        aug = augment_gaussian_noise(sample_clip, sigma=1.0, rng=np.random.default_rng(0))
        assert aug.shape == sample_clip.shape

    def test_values_differ(self, sample_clip):
        aug = augment_gaussian_noise(sample_clip, sigma=1.0, rng=np.random.default_rng(0))
        assert not np.allclose(aug, sample_clip)

    def test_zero_sigma_unchanged(self, sample_clip):
        aug = augment_gaussian_noise(sample_clip, sigma=0.0, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(aug, sample_clip)


class TestHorizontalFlip:
    def test_shape_preserved(self, sample_clip):
        aug = augment_horizontal_flip(sample_clip)
        assert aug.shape == sample_clip.shape

    def test_spatial_flip(self, sample_clip):
        aug = augment_horizontal_flip(sample_clip)
        # Spatial dimension (axis=2) is reversed
        np.testing.assert_array_equal(aug[:, :, :, 1], sample_clip[:, :, ::-1, 1])

    def test_u_component_negated(self, sample_clip):
        aug = augment_horizontal_flip(sample_clip)
        # u (horizontal displacement, channel 0) is negated
        np.testing.assert_array_equal(aug[:, :, :, 0], -sample_clip[:, :, ::-1, 0])

    def test_double_flip_identity(self, sample_clip):
        double = augment_horizontal_flip(augment_horizontal_flip(sample_clip))
        np.testing.assert_allclose(double, sample_clip)


class TestTemporalReverse:
    def test_shape_preserved(self, sample_clip):
        aug = augment_temporal_reverse(sample_clip)
        assert aug.shape == sample_clip.shape

    def test_temporal_order_reversed(self, sample_clip):
        aug = augment_temporal_reverse(sample_clip)
        # Reversed time, negated flow (opposite direction)
        np.testing.assert_array_equal(aug, -sample_clip[::-1])

    def test_double_reverse_identity(self, sample_clip):
        double = augment_temporal_reverse(augment_temporal_reverse(sample_clip))
        np.testing.assert_allclose(double, sample_clip)


class TestMagnitudeScale:
    def test_shape_preserved(self, sample_clip):
        aug = augment_magnitude_scale(sample_clip, scale=1.2)
        assert aug.shape == sample_clip.shape

    def test_scaling_correct(self, sample_clip):
        aug = augment_magnitude_scale(sample_clip, scale=0.8)
        np.testing.assert_allclose(aug, sample_clip * 0.8)

    def test_scale_one_unchanged(self, sample_clip):
        aug = augment_magnitude_scale(sample_clip, scale=1.0)
        np.testing.assert_array_equal(aug, sample_clip)


class TestApplyAugmentations:
    def test_returns_original_plus_augmented(self, sample_clip):
        augmented = apply_augmentations(sample_clip, rng=np.random.default_rng(0))
        # Should return at least the original clip
        assert len(augmented) >= 1
        # First element is always the original
        np.testing.assert_array_equal(augmented[0], sample_clip)

    def test_all_have_correct_shape(self, sample_clip):
        augmented = apply_augmentations(sample_clip, rng=np.random.default_rng(0))
        for aug in augmented:
            assert aug.shape == sample_clip.shape
