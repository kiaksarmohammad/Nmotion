"""
NeonatalRubric dataclass — the final output of the RL pipeline.

Contains:
- Selected feature names
- Per-feature thresholds + 95% CI
- Per-feature weights
- SHAP attributions
- Clinical direction metadata

Export: JSON artifact for clinical review.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NeonatalRubric dataclass
# ---------------------------------------------------------------------------

@dataclass
class NeonatalRubric:
    """
    Represents a discovered neonatal movement grading rubric.

    Attributes:
        selected_features: List of biomarker names included in the rubric.
        thresholds: {feature_name: threshold_value}
        weights: {feature_name: weight ∈ [0, 1]}
        threshold_ci: {feature_name: (ci_low, ci_high)} — bootstrap 95% CI
        metadata: Arbitrary metadata (pipeline version, timestamps, etc.)
        auroc: Final test AUROC achieved by the rubric.
        clinical_directions: {feature_name: direction} for interpretability.
    """

    selected_features: List[str] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    threshold_ci: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    auroc: float = 0.0
    clinical_directions: Dict[str, str] = field(default_factory=dict)

    def weighted_score(self, feature_means: Dict[str, float]) -> float:
        """
        Compute the rubric score for a single observation given per-feature means.
        Returns float in [0, 1].
        """
        score = 0.0
        w_total = sum(self.weights.values())
        if w_total == 0:
            return 0.0

        for feat in self.selected_features:
            thresh = self.thresholds.get(feat, 0.0)
            w = self.weights.get(feat, 0.0)
            val = feature_means.get(feat, 0.0)
            score += (w / w_total) * float(val > thresh)

        return float(np.clip(score, 0.0, 1.0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_features": self.selected_features,
            "thresholds": self.thresholds,
            "weights": self.weights,
            "threshold_ci": {
                k: list(v) for k, v in self.threshold_ci.items()
            },
            "metadata": self.metadata,
            "auroc": self.auroc,
            "clinical_directions": self.clinical_directions,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NeonatalRubric":
        r = cls(
            selected_features=d.get("selected_features", []),
            thresholds=d.get("thresholds", {}),
            weights=d.get("weights", {}),
            metadata=d.get("metadata", {}),
            auroc=d.get("auroc", 0.0),
            clinical_directions=d.get("clinical_directions", {}),
        )
        r.threshold_ci = {
            k: tuple(v) for k, v in d.get("threshold_ci", {}).items()
        }
        return r

    def __str__(self) -> str:
        lines = ["NeonatalRubric", "=" * 40]
        lines.append(f"AUROC: {self.auroc:.4f}")
        lines.append(f"Features ({len(self.selected_features)}): "
                     f"{', '.join(self.selected_features)}")
        lines.append("")
        lines.append(f"{'Feature':<20} {'Threshold':>12} {'CI':>20} {'Weight':>8} {'Direction':>12}")
        lines.append("-" * 75)
        for feat in self.selected_features:
            thresh = self.thresholds.get(feat, 0.0)
            ci = self.threshold_ci.get(feat, (float("nan"), float("nan")))
            w = self.weights.get(feat, 0.0)
            d = self.clinical_directions.get(feat, "?")
            lines.append(
                f"{feat:<20} {thresh:>12.4f} ({ci[0]:>8.4f}, {ci[1]:>8.4f})"
                f" {w:>8.4f} {d:>12}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# apply_rubric
# ---------------------------------------------------------------------------

def apply_rubric(rubric: NeonatalRubric, df: Any) -> float:
    """
    Apply rubric to a single DataFrame recording.
    Returns score in [0, 1].
    """
    feature_means = {
        feat: float(df[feat].mean())
        for feat in rubric.selected_features
        if feat in df.columns
    }
    return rubric.weighted_score(feature_means)


# ---------------------------------------------------------------------------
# SHAP attribution
# ---------------------------------------------------------------------------

def compute_shap_attribution(
    rubric: NeonatalRubric,
    windows: np.ndarray,
    labels: np.ndarray,
    biomarker_names: List[str],
    cfg=None,
    output_dir: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """
    Compute SHAP values for the rubric using KernelExplainer.
    Returns SHAP values [N, S] or None if shap not installed.
    Also saves a bar chart to output_dir if provided.
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — skipping SHAP attribution. pip install shap")
        return None

    selected_indices = [
        biomarker_names.index(f)
        for f in rubric.selected_features
        if f in biomarker_names
    ]

    if not selected_indices:
        return None

    # Extract per-window mean features [N, S]
    X = np.stack(
        [windows[:, :, i].mean(axis=1) for i in selected_indices], axis=1
    ).astype(np.float64)

    # Prediction function: rubric score
    def _predict(X_in: np.ndarray) -> np.ndarray:
        out = np.zeros(len(X_in))
        for i, feat in enumerate(rubric.selected_features):
            thresh = rubric.thresholds.get(feat, 0.0)
            w = rubric.weights.get(feat, 0.0)
            out += w * (X_in[:, i] > thresh).astype(float)
        w_sum = sum(rubric.weights.values())
        if w_sum > 0:
            out /= w_sum
        return out

    # Use a small background dataset
    n_bg = min(50, len(X))
    bg_idx = np.random.default_rng(42).choice(len(X), size=n_bg, replace=False)
    background = X[bg_idx]

    try:
        explainer = shap.KernelExplainer(_predict, background)
        # Explain a sample (limit for speed)
        n_explain = min(200, len(X))
        shap_values = explainer.shap_values(X[:n_explain], nsamples=100)
    except Exception as e:
        logger.warning("SHAP computation failed: %s", e)
        return None

    # Save bar chart
    if output_dir is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            feat_names = rubric.selected_features

            fig, ax = plt.subplots(figsize=(8, max(4, len(feat_names) * 0.5)))
            idx = np.argsort(mean_abs_shap)
            ax.barh(
                [feat_names[i] for i in idx],
                mean_abs_shap[idx],
                color="steelblue",
            )
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title("SHAP Feature Attribution — NeonatalRubric")
            plt.tight_layout()

            chart_path = Path(output_dir) / "shap_attribution.png"
            fig.savefig(chart_path, dpi=150)
            plt.close(fig)
            logger.info("SHAP chart saved to %s", chart_path)
        except Exception as e:
            logger.warning("Could not save SHAP chart: %s", e)

    return shap_values


# ---------------------------------------------------------------------------
# Supervised baseline (fallback if RL fails)
# ---------------------------------------------------------------------------

def fit_supervised_baseline(
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    test_windows: np.ndarray,
    test_labels: np.ndarray,
    biomarker_names: List[str],
    cfg,
    mlflow_run=None,
) -> Tuple[float, Any]:
    """
    Fit a Random Forest classifier as a supervised fallback.
    Returns (test_auroc, fitted_model).
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
    except ImportError:
        logger.error("scikit-learn required for supervised baseline.")
        return 0.0, None

    # Mean over time window per feature
    X_train = train_windows.mean(axis=1)  # [N, C]
    X_test = test_windows.mean(axis=1)
    y_train = (train_labels > 0).astype(int)
    y_test = (test_labels > 0).astype(int)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    probs = rf.predict_proba(X_test)[:, 1]
    try:
        auroc = float(roc_auc_score(y_test, probs))
    except ValueError:
        auroc = 0.5

    logger.info("Supervised RF baseline AUROC: %.4f", auroc)

    if mlflow_run is not None:
        try:
            import mlflow
            mlflow.log_metric("baseline_rf_auroc", auroc)
        except Exception:
            pass

    return auroc, rf


# ---------------------------------------------------------------------------
# Report export
# ---------------------------------------------------------------------------

def export_rubric_report(
    rubric: NeonatalRubric,
    validation_results: Dict[str, Any],
    shap_values: Optional[np.ndarray],
    path: Path,
    mlflow_run=None,
) -> None:
    """
    Export human-readable JSON rubric report to path.
    Logs as MLflow artifact if run provided.
    """
    shap_summary = None
    if shap_values is not None:
        mean_abs = np.abs(shap_values).mean(axis=0).tolist()
        shap_summary = dict(zip(rubric.selected_features, mean_abs))

    report = {
        "rubric": rubric.to_dict(),
        "validation": validation_results,
        "shap_attribution": shap_summary,
        "clinical_interpretation": {
            feat: {
                "threshold": rubric.thresholds.get(feat),
                "weight": rubric.weights.get(feat),
                "direction": rubric.clinical_directions.get(feat),
                "ci_95": list(rubric.threshold_ci.get(feat, (None, None))),
                "shap_importance": (
                    shap_summary.get(feat) if shap_summary else None
                ),
            }
            for feat in rubric.selected_features
        },
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Rubric report exported to %s", path)

    if mlflow_run is not None:
        try:
            import mlflow
            mlflow.log_artifact(str(path), artifact_path="reports")
        except Exception as e:
            logger.warning("Could not log rubric artifact: %s", e)
