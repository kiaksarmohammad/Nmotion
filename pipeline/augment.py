"""Flow field augmentation for temporal clips.

All augmentations operate directly on flow fields [N, H, W, 2] where
channels are (u, v) displacement vectors. This avoids re-running RAFT
on augmented video frames.

Key invariant: augmentations must preserve the physical meaning of flow
vectors. Spatial flips require negating the corresponding displacement
component. Temporal reversal negates both components (reversed motion).
"""

from __future__ import annotations

from typing import List

import numpy as np


def augment_gaussian_noise(
    clip: np.ndarray, sigma: float = 1.0, rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add Gaussian noise to flow vectors.

    Args:
        clip: [N, H, W, 2] flow clip.
        sigma: Noise standard deviation in pixels.
        rng: Random generator for reproducibility.
    """
    if sigma == 0.0:
        return clip.copy()
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0, sigma, size=clip.shape).astype(clip.dtype)
    return clip + noise


def augment_horizontal_flip(clip: np.ndarray) -> np.ndarray:
    """Flip flow field horizontally.

    Reverses the spatial width axis and negates the u (horizontal)
    displacement component, since flipped motion points the other way.
    """
    flipped = clip[:, :, ::-1, :].copy()
    flipped[..., 0] *= -1  # negate u
    return flipped


def augment_temporal_reverse(clip: np.ndarray) -> np.ndarray:
    """Reverse temporal order and negate flow vectors.

    Playing motion backwards reverses displacement direction.
    """
    return -clip[::-1].copy()


def augment_magnitude_scale(clip: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Scale flow magnitude uniformly.

    Simulates faster/slower movement without changing direction.
    """
    return clip * scale


def apply_augmentations(
    clip: np.ndarray,
    noise_sigma: float = 1.0,
    scale_range: tuple[float, float] = (0.8, 1.2),
    rng: np.random.Generator | None = None,
) -> List[np.ndarray]:
    """Apply a fixed augmentation pipeline to one clip.

    Returns the original clip plus augmented versions:
    - Original (always first)
    - Horizontal flip
    - Temporal reverse
    - Gaussian noise (1 variant)
    - Magnitude scale (1 variant, random within range)

    Args:
        clip: [N, H, W, 2] flow clip.
        noise_sigma: Gaussian noise sigma.
        scale_range: (min, max) for random magnitude scaling.
        rng: Random generator.

    Returns:
        List of [N, H, W, 2] arrays (original + 4 augmented = 5 total).
    """
    if rng is None:
        rng = np.random.default_rng()

    augmented = [clip]  # original always first
    augmented.append(augment_horizontal_flip(clip))
    augmented.append(augment_temporal_reverse(clip))
    augmented.append(augment_gaussian_noise(clip, sigma=noise_sigma, rng=rng))

    scale = rng.uniform(scale_range[0], scale_range[1])
    augmented.append(augment_magnitude_scale(clip, scale=scale))

    return augmented
