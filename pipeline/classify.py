"""XGBoost classification with grouped cross-validation.

Clips from the same video are kept in the same fold (via
``StratifiedGroupKFold``) to prevent data leakage. Class weights are
inversely proportional to frequency, giving underrepresented classes
larger sample weights.

``train_evaluate_grouped_cv`` is the single CV entry point — it's
called from:
  * `run.py` (clip-level classification stage)
  * `scripts/classify_discovered.py` (feature-set experiments)

The discovery script just supplies its own ``feature_cols`` /
``group_col`` / ``experiment_name`` and unpacks the same result dict.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# Baseline feature set — matches column names emitted by
# pipeline.feature_battery.compute_all_features.
FEATURE_COLS = [
    "sample_entropy",
    "spectral_entropy",
    "dfa_alpha",
    "symmetry_mean",
    "peak_frequency",
    "ke_mean",
    "ke_std",
    "flow_mean",
    "flow_std",
    "flow_skew",
    "flow_kurtosis",
]
MSE_PREFIX = "mse_scale_"


def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Baseline FEATURE_COLS present in df, plus any mse_scale_* columns."""
    mse_cols = [c for c in df.columns if c.startswith(MSE_PREFIX)]
    present = [c for c in FEATURE_COLS if c in df.columns]
    return present + mse_cols


def compute_class_weights(labels: pd.Series) -> Dict[str, float]:
    """Inverse-frequency class weights: w_c = N / (K · N_c)."""
    counts = labels.value_counts()
    n_total = len(labels)
    n_classes = len(counts)
    return {
        cls: n_total / (n_classes * count)
        for cls, count in counts.items()
    }


def train_evaluate_grouped_cv(
    df: pd.DataFrame,
    n_folds: int = 5,
    feature_cols: Optional[List[str]] = None,
    group_col: str = "video_id",
    experiment_name: str = "cv",
    return_importances: bool = False,
) -> Dict:
    """Train XGBoost with StratifiedGroupKFold cross-validation.

    Args:
        df: Feature DataFrame; must contain ``group_col`` and a ``group`` column.
        n_folds: Desired number of CV folds. Automatically reduced to
            ``min(n_folds, min_class_count)`` so sklearn's stratification
            constraint is satisfied.
        feature_cols: Feature column names. If ``None``, uses baseline +
            auto-discovered ``mse_scale_*`` columns.
        group_col: Column whose unique values define CV groups (default
            ``video_id``; discovery script uses ``video``).
        experiment_name: Label for log messages — distinguishes concurrent
            experiments in shared log output.
        return_importances: If True, include ``feature_importances`` (a
            sorted DataFrame) in the result dict.

    Returns:
        Dict with keys: ``accuracy``, ``accuracy_std``, ``weighted_f1``,
        ``f1_std``, ``fold_accuracies``, ``fold_f1s``, ``per_class``,
        ``confusion_matrix``, ``class_names``, ``class_weights``,
        ``n_features``, ``n_folds``, and — if requested —
        ``feature_importances``.
    """
    if feature_cols is None:
        feature_cols = _get_feature_cols(df)

    valid_cols = [c for c in feature_cols if c in df.columns]
    if not valid_cols:
        logger.warning("[%s] No valid feature columns", experiment_name)
        return {}

    # Drop rows where every feature is NaN
    X_df = df[valid_cols]
    valid_mask = X_df.notna().any(axis=1)
    df_clean = df[valid_mask].copy()
    X = df_clean[valid_cols].fillna(0).values

    le = LabelEncoder()
    y = le.fit_transform(df_clean["group"])
    groups = df_clean[group_col].values

    class_weight_map = compute_class_weights(df_clean["group"])
    sample_weights = np.array([class_weight_map[g] for g in df_clean["group"]])

    counts = df_clean["group"].value_counts()
    min_class_count = int(counts.min())
    actual_folds = min(n_folds, min_class_count)
    if actual_folds < 2:
        logger.warning(
            "[%s] Smallest class has %d samples — forcing n_folds=2",
            experiment_name, min_class_count,
        )
        actual_folds = 2

    sgkf = StratifiedGroupKFold(
        n_splits=actual_folds, shuffle=True, random_state=42,
    )

    all_preds = np.full(len(df_clean), -1, dtype=int)
    fold_accs: List[float] = []
    fold_f1s: List[float] = []
    importances_sum = np.zeros(len(valid_cols))

    for fold_idx, (train_idx, test_idx) in enumerate(
        sgkf.split(X, y, groups)
    ):
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )
        model.fit(
            X[train_idx], y[train_idx],
            sample_weight=sample_weights[train_idx],
        )
        preds = model.predict(X[test_idx])
        all_preds[test_idx] = preds

        fold_acc = accuracy_score(y[test_idx], preds)
        fold_f1 = f1_score(y[test_idx], preds, average="weighted")
        fold_accs.append(fold_acc)
        fold_f1s.append(fold_f1)
        importances_sum += model.feature_importances_

        logger.info(
            "[%s] Fold %d: acc=%.3f  f1=%.3f",
            experiment_name, fold_idx + 1, fold_acc, fold_f1,
        )

    valid = all_preds >= 0
    y_true = y[valid]
    y_pred = all_preds[valid]
    class_names = list(le.classes_)

    overall_acc = accuracy_score(y_true, y_pred)
    overall_f1 = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True,
    )

    logger.info(
        "[%s] Overall: acc=%.3f (±%.3f)  f1=%.3f (±%.3f)",
        experiment_name,
        overall_acc, np.std(fold_accs),
        overall_f1, np.std(fold_f1s),
    )
    logger.info(
        "[%s]\n%s",
        experiment_name,
        classification_report(y_true, y_pred, target_names=class_names),
    )

    result: Dict = {
        "name": experiment_name,
        "n_features": len(valid_cols),
        "n_folds": actual_folds,
        "accuracy": overall_acc,
        "accuracy_std": float(np.std(fold_accs)),
        "weighted_f1": overall_f1,
        "f1_std": float(np.std(fold_f1s)),
        "fold_accuracies": fold_accs,
        "fold_f1s": fold_f1s,
        "per_class": report,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "class_weights": class_weight_map,
    }

    if return_importances:
        avg_importance = importances_sum / actual_folds
        result["feature_importances"] = pd.DataFrame({
            "feature": valid_cols,
            "importance": avg_importance,
        }).sort_values("importance", ascending=False)

    return result


def aggregate_clip_predictions(clip_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate clip-level predictions to video-level by majority vote.

    Args:
        clip_df: DataFrame with 'video_id', 'predicted', 'true_label' columns.

    Returns:
        DataFrame with one row per video: video_id, predicted, true_label, n_clips.
    """
    rows = []
    for vid_id, group in clip_df.groupby("video_id"):
        predicted = group["predicted"].mode().iloc[0]
        true_label = group["true_label"].iloc[0]
        rows.append({
            "video_id": vid_id,
            "predicted": predicted,
            "true_label": true_label,
            "n_clips": len(group),
        })
    return pd.DataFrame(rows)
