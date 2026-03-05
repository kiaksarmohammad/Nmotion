"""
Composite reward function for the RL pipeline.

R_total = R_terminal + λ₁·R_intermediate + λ₂·R_complexity + λ₃·R_plausibility

Terminal rewards:
  True positive  → +1.0
  False positive → -0.5  (× fp_weight)
  False negative → -3.0  (× fn_weight)

Plausibility reward:
  ±0.2 per feature based on clinical prior directions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Clinical priors (default; overridden by PipelineConfig at runtime)
# ---------------------------------------------------------------------------

CLINICAL_PRIORS: Dict[str, str] = {
    "entropy":         "down",
    "fractal_dim":     "down",
    "kinetic_energy":  "up",
    "root_stress":     "up",
    "phase_x":         "low_variance",
    "phase_v":         "low_variance",
    "mean_velocity":   "up",
    "peak_frequency":  "up",
    "angular_jerk":    "up",
}


class CompositeReward:
    """
    Computes all reward components and returns a weighted total.
    All methods are pure (no side effects).
    """

    def __init__(
        self,
        cfg=None,
        biomarker_names: Optional[List[str]] = None,
        clinical_directions: Optional[Dict[str, str]] = None,
    ):
        self.lambda1 = cfg.LAMBDA1 if cfg else 0.4
        self.lambda2 = cfg.LAMBDA2 if cfg else 0.1
        self.lambda3 = cfg.LAMBDA3 if cfg else 0.2
        self.fn_weight = cfg.FN_WEIGHT if cfg else 3.0
        self.fp_weight = cfg.FP_WEIGHT if cfg else 1.0
        self.biomarker_names = biomarker_names or list(CLINICAL_PRIORS.keys())
        self.directions = clinical_directions or CLINICAL_PRIORS.copy()

    # ------------------------------------------------------------------
    # Terminal reward
    # ------------------------------------------------------------------

    def terminal(
        self,
        y_true: int,
        y_pred: int,
    ) -> float:
        """
        Single-sample terminal reward.
        y_true / y_pred: 0 = normal, 1 = pathological.
        """
        if y_true == 1 and y_pred == 1:
            return 1.0                          # True positive
        if y_true == 0 and y_pred == 0:
            return 1.0                          # True negative
        if y_true == 1 and y_pred == 0:
            return -1.0 * self.fn_weight        # False negative → heavy penalty
        # False positive
        return -0.5 * self.fp_weight

    def terminal_batch(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Mean terminal reward over a batch."""
        rewards = np.array(
            [self.terminal(int(yt), int(yp)) for yt, yp in zip(y_true, y_pred)]
        )
        return float(rewards.mean())

    # ------------------------------------------------------------------
    # Intermediate reward — ΔAUROC
    # ------------------------------------------------------------------

    def intermediate(
        self,
        rubric_scores: np.ndarray,
        val_labels: np.ndarray,
        prev_auroc: float,
    ) -> float:
        """
        Returns ΔAUROC = current_AUROC - prev_AUROC.
        rubric_scores: continuous scores [N] for val set.
        """
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            return 0.0

        y_binary = (val_labels > 0).astype(int)
        if len(np.unique(y_binary)) < 2:
            return 0.0

        try:
            current_auroc = roc_auc_score(y_binary, rubric_scores)
        except ValueError:
            return 0.0

        return float(current_auroc - prev_auroc)

    # ------------------------------------------------------------------
    # Complexity reward
    # ------------------------------------------------------------------

    def complexity(self, n_selected: int) -> float:
        """Penalize large feature sets: -0.1 per selected feature."""
        return -0.1 * n_selected

    # ------------------------------------------------------------------
    # Plausibility reward
    # ------------------------------------------------------------------

    def plausibility(
        self,
        thresholds: Dict[str, float],
        data_means: Dict[str, float],
        data_vars: Dict[str, float],
    ) -> float:
        """
        +0.2 per feature that satisfies clinical prior direction,
        -0.2 per feature that violates it.

        Directions:
          "up"           → threshold > data_mean (pathological = high values)
          "down"         → threshold < data_mean (pathological = low values)
          "low_variance" → data_var should be low (< global median var)
        """
        score = 0.0
        median_var = np.median(list(data_vars.values())) if data_vars else 1.0

        for feat, thresh in thresholds.items():
            direction = self.directions.get(feat)
            if direction is None:
                continue
            mean = data_means.get(feat, 0.0)
            var = data_vars.get(feat, 1.0)

            if direction == "up":
                score += 0.2 if thresh > mean else -0.2
            elif direction == "down":
                score += 0.2 if thresh < mean else -0.2
            elif direction == "low_variance":
                score += 0.2 if var < median_var else -0.2

        return score

    # ------------------------------------------------------------------
    # Total reward
    # ------------------------------------------------------------------

    def total(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        rubric_scores: np.ndarray,
        val_labels: np.ndarray,
        prev_auroc: float,
        n_selected: int,
        thresholds: Dict[str, float],
        data_means: Dict[str, float],
        data_vars: Dict[str, float],
    ) -> float:
        """
        R_total = R_terminal + λ₁·R_int + λ₂·R_comp + λ₃·R_plaus
        """
        r_term = self.terminal_batch(y_true, y_pred)
        r_int = self.intermediate(rubric_scores, val_labels, prev_auroc)
        r_comp = self.complexity(n_selected)
        r_plaus = self.plausibility(thresholds, data_means, data_vars)

        total = (
            r_term
            + self.lambda1 * r_int
            + self.lambda2 * r_comp
            + self.lambda3 * r_plaus
        )
        return float(total)


# ---------------------------------------------------------------------------
# Utility: check plausibility violations for validation
# ---------------------------------------------------------------------------

def check_plausibility_violations(
    thresholds: Dict[str, float],
    data_means: Dict[str, float],
    directions: Optional[Dict[str, str]] = None,
) -> List[str]:
    """
    Returns list of feature names whose threshold violates clinical prior.
    Used by validation.py for the plausibility audit.
    """
    if directions is None:
        directions = CLINICAL_PRIORS

    violations = []
    for feat, thresh in thresholds.items():
        direction = directions.get(feat)
        mean = data_means.get(feat, 0.0)
        if direction == "up" and thresh <= mean:
            violations.append(feat)
        elif direction == "down" and thresh >= mean:
            violations.append(feat)
    return violations
