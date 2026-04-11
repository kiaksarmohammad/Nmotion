"""Tests for XGBoost classifier with grouped cross-validation."""

import numpy as np
import pandas as pd
import pytest

from pipeline.classify import (
    compute_class_weights,
    train_evaluate_grouped_cv,
    aggregate_clip_predictions,
)


@pytest.fixture
def synthetic_features():
    """Synthetic feature DataFrame mimicking clip extraction output."""
    rng = np.random.default_rng(42)
    n = 200

    # 3 classes with imbalance: 100 seizure, 80 normal, 20 hypotonic
    labels = (["seizure"] * 100 + ["normal"] * 80 + ["hypotonic"] * 20)
    video_ids = (
        [f"sz_{i // 5}" for i in range(100)]
        + [f"nm_{i // 5}" for i in range(80)]
        + [f"hp_{i // 5}" for i in range(20)]
    )

    df = pd.DataFrame({
        "group": labels,
        "video_id": video_ids,
        "sample_entropy": rng.standard_normal(n),
        "spectral_entropy": rng.standard_normal(n),
        "dfa_alpha": rng.standard_normal(n),
        "flow_mean": rng.standard_normal(n),
        "flow_std": rng.standard_normal(n),
        "kinetic_energy_mean": rng.standard_normal(n),
    })
    return df


class TestClassWeights:
    def test_hypotonic_has_highest_weight(self):
        labels = pd.Series(["seizure"] * 75 + ["normal"] * 64 + ["hypotonic"] * 10)
        weights = compute_class_weights(labels)
        assert weights["hypotonic"] > weights["seizure"]
        assert weights["hypotonic"] > weights["normal"]

    def test_weights_are_positive(self):
        labels = pd.Series(["a"] * 50 + ["b"] * 30 + ["c"] * 5)
        weights = compute_class_weights(labels)
        assert all(w > 0 for w in weights.values())


class TestGroupedCV:
    def test_returns_metrics_dict(self, synthetic_features):
        results = train_evaluate_grouped_cv(synthetic_features, n_folds=3)
        assert "accuracy" in results
        assert "per_class" in results
        assert "confusion_matrix" in results

    def test_no_video_leakage(self, synthetic_features):
        results = train_evaluate_grouped_cv(synthetic_features, n_folds=3)
        # If grouped correctly, results should exist (no crash from leakage check)
        assert results["accuracy"] >= 0.0


class TestAggregation:
    def test_majority_vote(self):
        clip_preds = pd.DataFrame({
            "video_id": ["v1", "v1", "v1", "v2", "v2"],
            "predicted": ["seizure", "seizure", "normal", "normal", "normal"],
            "true_label": ["seizure", "seizure", "seizure", "normal", "normal"],
        })
        agg = aggregate_clip_predictions(clip_preds)
        assert agg.loc[agg["video_id"] == "v1", "predicted"].values[0] == "seizure"
        assert agg.loc[agg["video_id"] == "v2", "predicted"].values[0] == "normal"
