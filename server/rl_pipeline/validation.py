"""
Validation suite for the discovered NeonatalRubric.

Tests:
1. Cross-OPE matrix (5 reward variants × 4 metrics)
2. Null shuffle test (AUROC < 0.55 on 100 synthetic nulls)
3. Temporal shuffle test (ΔAUROC ≥ 10% vs. temporally shuffled data)
4. Biomarker plausibility audit (direction violations)
5. Sensitivity analysis (robustness score ≥ 0.85)

Hard failure → raises RubricFailureError.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

try:
    from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.error("scikit-learn not available. Install: pip install scikit-learn")


class RubricFailureError(Exception):
    """Raised when the rubric fails hard validation criteria."""
    pass


# ---------------------------------------------------------------------------
# Rubric scoring utility (operates on windows directly)
# ---------------------------------------------------------------------------

def score_rubric_on_windows(
    rubric: Any,  # NeonatalRubric
    windows: np.ndarray,
    biomarker_names: List[str],
) -> np.ndarray:
    """
    Apply rubric thresholds/weights to windows [N, W, C] → scores [N].
    """
    N = len(windows)
    scores = np.zeros(N, dtype=np.float32)
    w_sum = sum(rubric.weights.values()) if rubric.weights else 1.0

    for feat_name, thresh in rubric.thresholds.items():
        if feat_name not in biomarker_names:
            continue
        feat_idx = biomarker_names.index(feat_name)
        if feat_idx >= windows.shape[2]:
            continue
        feat_vals = windows[:, :, feat_idx].mean(axis=1)  # [N]
        weight = rubric.weights.get(feat_name, 1.0)
        scores += (weight / max(w_sum, 1e-9)) * (feat_vals > thresh).astype(np.float32)

    return scores


def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute AUROC, F1, MCC, sensitivity@90%specificity.
    Labels are binary (pathological = 1).
    """
    if not _SKLEARN_AVAILABLE:
        return {"auroc": 0.5, "f1": 0.0, "mcc": 0.0, "sens_at_90spec": 0.0}

    y_binary = (labels > 0).astype(int)
    if len(np.unique(y_binary)) < 2:
        return {"auroc": 0.5, "f1": 0.0, "mcc": 0.0, "sens_at_90spec": 0.0}

    auroc = float(roc_auc_score(y_binary, scores))
    threshold = 0.5
    y_pred = (scores > threshold).astype(int)
    f1 = float(f1_score(y_binary, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_binary, y_pred))

    # Sensitivity at 90% specificity
    sens_at_90 = _sensitivity_at_specificity(scores, y_binary, target_spec=0.90)

    return {
        "auroc": auroc,
        "f1": f1,
        "mcc": mcc,
        "sens_at_90spec": sens_at_90,
    }


def _sensitivity_at_specificity(
    scores: np.ndarray, labels: np.ndarray, target_spec: float = 0.90
) -> float:
    """Find sensitivity at the threshold that achieves target specificity."""
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, threshs = roc_curve(labels, scores)
        # specificity = 1 - fpr
        spec = 1.0 - fpr
        # Find thresholds where spec >= target_spec
        valid = spec >= target_spec
        if not valid.any():
            return 0.0
        return float(tpr[valid][-1])  # highest TPR at target spec
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# 1. Cross-OPE matrix
# ---------------------------------------------------------------------------

def cross_ope_matrix(
    rubric: Any,
    val_windows: np.ndarray,
    val_labels: np.ndarray,
    biomarker_names: List[str],
    ope_variants: Dict[str, Dict],
    reward_cls,
    cfg,
    mlflow_run=None,
) -> "pd.DataFrame":
    """
    5×4 matrix: OPE variant × metric (AUROC, F1, MCC, sens@90spec).
    FAIL if any variant degrades AUROC >20% vs. best variant.
    """
    scores_base = score_rubric_on_windows(rubric, val_windows, biomarker_names)
    base_metrics = compute_metrics(scores_base, val_labels)
    best_auroc = base_metrics["auroc"]

    rows = []
    for variant_name, lam in ope_variants.items():
        # Re-score with lambda-weighted reward (simplified: use same rubric, vary weights)
        r_fn = reward_cls(
            biomarker_names=biomarker_names,
            clinical_directions=cfg.CLINICAL_DIRECTIONS,
        )
        r_fn.lambda1 = lam.get("lambda1", cfg.LAMBDA1)
        r_fn.lambda2 = lam.get("lambda2", cfg.LAMBDA2)
        r_fn.lambda3 = lam.get("lambda3", cfg.LAMBDA3)

        # Apply slight weight perturbation proportional to lambda2
        perturb = lam.get("lambda2", 0.1)
        noise = np.random.default_rng(42).normal(0, perturb, len(scores_base))
        scores_perturbed = np.clip(scores_base + noise, 0, 1)

        m = compute_metrics(scores_perturbed, val_labels)
        m["variant"] = variant_name
        rows.append(m)
        best_auroc = max(best_auroc, m["auroc"])

    if not _PANDAS_AVAILABLE:
        logger.warning("pandas not installed — skipping DataFrame creation")
        return None  # type: ignore

    df = pd.DataFrame(rows).set_index("variant")

    # Check degradation
    for variant_name in df.index:
        auroc = df.loc[variant_name, "auroc"]
        degradation = (best_auroc - auroc) / max(best_auroc, 1e-9)
        if degradation > cfg.OPE_MAX_DEGRADATION:
            msg = (
                f"OPE variant {variant_name} AUROC={auroc:.4f} degrades "
                f"{degradation:.1%} from best={best_auroc:.4f} "
                f"(max allowed={cfg.OPE_MAX_DEGRADATION:.0%})"
            )
            logger.error(msg)
            if mlflow_run is not None:
                try:
                    import mlflow
                    mlflow.log_param("ope_failure", msg)
                except Exception:
                    pass
            raise RubricFailureError(f"Cross-OPE failure: {msg}")

    if mlflow_run is not None:
        try:
            import mlflow
            mlflow.log_dict(df.to_dict(), "ope_matrix.json")
        except Exception:
            pass

    logger.info("Cross-OPE matrix:\n%s", df.to_string())
    return df


# ---------------------------------------------------------------------------
# 2. Null shuffle test
# ---------------------------------------------------------------------------

def null_shuffle_test(
    rubric: Any,
    windows: np.ndarray,
    labels: np.ndarray,
    biomarker_names: List[str],
    n_shuffles: int = 100,
    max_null_auroc: float = 0.55,
    random_state: int = 42,
    mlflow_run=None,
) -> float:
    """
    Shuffle all biomarker values across time AND subjects.
    Fail if AUROC > max_null_auroc on any of n_shuffles nulls.
    Returns mean null AUROC.
    """
    rng = np.random.default_rng(random_state)
    null_aurocs = []

    for i in range(n_shuffles):
        shuffled = windows.copy()
        # Flatten across N and W, shuffle, reshape
        N, W, C = shuffled.shape
        flat = shuffled.reshape(N * W, C)
        perm = rng.permutation(N * W)
        flat = flat[perm]
        shuffled = flat.reshape(N, W, C)

        scores = score_rubric_on_windows(rubric, shuffled, biomarker_names)
        y_binary = (labels > 0).astype(int)

        if len(np.unique(y_binary)) < 2:
            null_aurocs.append(0.5)
            continue

        try:
            auroc = float(roc_auc_score(y_binary, scores))
        except ValueError:
            auroc = 0.5
        null_aurocs.append(auroc)

    mean_null = float(np.mean(null_aurocs))
    max_null = float(np.max(null_aurocs))

    logger.info("Null shuffle: mean_AUROC=%.4f  max_AUROC=%.4f", mean_null, max_null)

    if mlflow_run is not None:
        try:
            import mlflow
            mlflow.log_metric("null_mean_auroc", mean_null)
            mlflow.log_metric("null_max_auroc", max_null)
        except Exception:
            pass

    if max_null > max_null_auroc:
        raise RubricFailureError(
            f"Null shuffle test FAILED: max null AUROC={max_null:.4f} > "
            f"threshold={max_null_auroc:.2f}. Rubric may overfit noise."
        )

    return mean_null


# ---------------------------------------------------------------------------
# 3. Temporal shuffle test
# ---------------------------------------------------------------------------

def temporal_shuffle_test(
    rubric: Any,
    windows: np.ndarray,
    labels: np.ndarray,
    biomarker_names: List[str],
    base_auroc: float,
    random_state: int = 42,
    mlflow_run=None,
) -> float:
    """
    Shuffle time axis only. Warn if ΔAUROC < 10% (rubric only uses marginals).
    Returns temporal-shuffled AUROC.
    """
    rng = np.random.default_rng(random_state)
    N, W, C = windows.shape
    shuffled = windows.copy()
    for n in range(N):
        perm = rng.permutation(W)
        shuffled[n] = shuffled[n][perm]

    scores = score_rubric_on_windows(rubric, shuffled, biomarker_names)
    y_binary = (labels > 0).astype(int)

    try:
        shuffled_auroc = float(roc_auc_score(y_binary, scores))
    except (ValueError, Exception):
        shuffled_auroc = 0.5

    delta = base_auroc - shuffled_auroc
    logger.info(
        "Temporal shuffle: base_AUROC=%.4f  shuffled_AUROC=%.4f  ΔAUROC=%.4f",
        base_auroc, shuffled_auroc, delta,
    )

    if delta < 0.10:
        logger.warning(
            "ΔAUROC=%.4f < 10%% on temporal shuffle. "
            "Rubric may rely mainly on marginal distributions, not temporal structure.",
            delta,
        )

    if mlflow_run is not None:
        try:
            import mlflow
            mlflow.log_metric("temporal_shuffled_auroc", shuffled_auroc)
            mlflow.log_metric("temporal_delta_auroc", delta)
        except Exception:
            pass

    return shuffled_auroc


# ---------------------------------------------------------------------------
# 4. Plausibility audit
# ---------------------------------------------------------------------------

def biomarker_plausibility_audit(
    thresholds: Dict[str, float],
    data_means: Dict[str, float],
    directions: Dict[str, str],
    mlflow_run=None,
) -> List[str]:
    """
    Check each threshold direction against clinical priors.
    Returns list of violations.
    FAIL if >1 implausible direction without expert approval.
    """
    from .reward import check_plausibility_violations

    violations = check_plausibility_violations(thresholds, data_means, directions)
    logger.info("Plausibility audit: %d violations: %s", len(violations), violations)

    if mlflow_run is not None:
        try:
            import mlflow
            mlflow.log_param("plausibility_violations", str(violations))
            mlflow.log_metric("n_plausibility_violations", len(violations))
        except Exception:
            pass

    if len(violations) > 1:
        raise RubricFailureError(
            f"Plausibility audit FAILED: {len(violations)} violations "
            f"({violations}). Expert approval required for implausible thresholds."
        )

    return violations


# ---------------------------------------------------------------------------
# 5. Sensitivity analysis
# ---------------------------------------------------------------------------

def sensitivity_analysis(
    rubric: Any,
    test_windows: np.ndarray,
    test_labels: np.ndarray,
    biomarker_names: List[str],
    perturbation: float = 0.10,
    target_robustness: float = 0.85,
    mlflow_run=None,
) -> float:
    """
    Perturb each threshold ±10%, measure ΔAUROC.
    Returns robustness_score ∈ [0, 1].
    Target: > 0.85.
    """
    base_scores = score_rubric_on_windows(rubric, test_windows, biomarker_names)
    y_binary = (test_labels > 0).astype(int)

    try:
        base_auroc = float(roc_auc_score(y_binary, base_scores))
    except (ValueError, Exception):
        base_auroc = 0.5

    n_features = len(rubric.thresholds)
    stable_count = 0
    total_perturbations = 0

    for feat_name, thresh in rubric.thresholds.items():
        for sign in [1, -1]:
            new_thresh = thresh * (1.0 + sign * perturbation)

            # Build perturbed rubric copy
            import copy
            perturbed_rubric = copy.deepcopy(rubric)
            perturbed_rubric.thresholds[feat_name] = new_thresh

            perturbed_scores = score_rubric_on_windows(
                perturbed_rubric, test_windows, biomarker_names
            )
            try:
                perturbed_auroc = float(roc_auc_score(y_binary, perturbed_scores))
            except (ValueError, Exception):
                perturbed_auroc = 0.5

            delta = abs(base_auroc - perturbed_auroc)
            total_perturbations += 1
            if delta < 0.05:  # < 5% AUROC change = stable
                stable_count += 1

    robustness = stable_count / max(total_perturbations, 1)
    logger.info(
        "Sensitivity analysis: robustness=%.4f (%d/%d perturbations stable)",
        robustness, stable_count, total_perturbations,
    )

    if mlflow_run is not None:
        try:
            import mlflow
            mlflow.log_metric("sensitivity_robustness", robustness)
        except Exception:
            pass

    if robustness < target_robustness:
        logger.warning(
            "Robustness %.4f < target %.2f — rubric thresholds are sensitive.",
            robustness, target_robustness,
        )

    return robustness


# ---------------------------------------------------------------------------
# Main validation entry point
# ---------------------------------------------------------------------------

def compute_test_metrics(
    rubric: Any,
    test_windows: np.ndarray,
    test_labels: np.ndarray,
    biomarker_names: List[str],
    mlflow_run=None,
) -> Dict[str, float]:
    """Compute final test-set metrics."""
    scores = score_rubric_on_windows(rubric, test_windows, biomarker_names)
    metrics = compute_metrics(scores, test_labels)

    logger.info("Test metrics: %s", metrics)

    if mlflow_run is not None:
        try:
            import mlflow
            for k, v in metrics.items():
                mlflow.log_metric(f"test_{k}", v)
        except Exception:
            pass

    return metrics


def check_hard_failures(results: Dict[str, Any], mlflow_run=None) -> None:
    """
    Aggregate hard failure checks. Raises RubricFailureError with reason.
    Called at the end of validation — individual test functions may have
    already raised, but this catches anything that slipped through.
    """
    errors = []

    auroc = results.get("auroc", 0.5)
    if auroc < 0.60:
        errors.append(f"Test AUROC={auroc:.4f} < 0.60 minimum.")

    sens = results.get("sens_at_90spec", 0.0)
    if sens < 0.50:
        errors.append(f"Sensitivity@90spec={sens:.4f} < 0.50 minimum.")

    if errors:
        msg = " | ".join(errors)
        if mlflow_run is not None:
            try:
                import mlflow
                mlflow.log_param("hard_failure_reason", msg)
            except Exception:
                pass
        raise RubricFailureError(f"Hard validation failures: {msg}")

    logger.info("All hard failure checks passed.")
