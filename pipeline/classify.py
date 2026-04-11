"""XGBoost classification with grouped cross-validation.

Clips from the same video are kept in the same fold to prevent data
leakage. Class weights are inversely proportional to frequency,
giving the underrepresented hypotonic class ~7.5x the weight.
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
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# Features used for classification — must match columns from features.py
FEATURE_COLS = [
    "sample_entropy",
    "spectral_entropy",
    "dfa_alpha",
    "symmetry_index",
    "peak_frequency",
    "kinetic_energy_mean",
    "kinetic_energy_std",
    "flow_mean",
    "flow_std",
    "flow_skew",
    "flow_kurtosis",
]
# Include MSE columns dynamically
MSE_PREFIX = "mse_scale_"


def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return feature column names present in the DataFrame."""
    mse_cols = [c for c in df.columns if c.startswith(MSE_PREFIX)]
    present = [c for c in FEATURE_COLS if c in df.columns]
    return present + mse_cols


def compute_class_weights(labels: pd.Series) -> Dict[str, float]:
    """Compute inverse-frequency class weights.

    weight_c = N_total / (N_classes * N_c)

    For 75:64:10 → seizure=0.66, normal=0.78, hypotonic=4.97
    """
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
) -> Dict:
    """Train XGBoost with StratifiedGroupKFold cross-validation.

    Groups by video_id so clips from the same video never span folds.

    Args:
        df: Feature DataFrame with 'group', 'video_id', and feature columns.
        n_folds: Number of CV folds.
        feature_cols: Override feature columns. Auto-detected if None.

    Returns:
        Dict with 'accuracy', 'per_class' (classification report dict),
        'confusion_matrix', and 'fold_accuracies'.
    """
    if feature_cols is None:
        feature_cols = _get_feature_cols(df)

    le = LabelEncoder()
    y = le.fit_transform(df["group"])
    X = df[feature_cols].fillna(0).values
    groups = df["video_id"].values

    class_weights = compute_class_weights(df["group"])
    sample_weights = np.array([class_weights[g] for g in df["group"]])

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_preds = np.full(len(df), -1, dtype=int)
    fold_accs: List[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="mlogloss",
            random_state=42,
        )
        model.fit(
            X[train_idx], y[train_idx],
            sample_weight=sample_weights[train_idx],
        )
        preds = model.predict(X[test_idx])
        all_preds[test_idx] = preds

        fold_acc = accuracy_score(y[test_idx], preds)
        fold_accs.append(fold_acc)
        logger.info("Fold %d: accuracy=%.3f", fold_idx + 1, fold_acc)

    # Overall metrics
    valid_mask = all_preds >= 0
    y_true = y[valid_mask]
    y_pred = all_preds[valid_mask]

    class_names = list(le.classes_)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    overall_acc = accuracy_score(y_true, y_pred)
    logger.info("Overall accuracy: %.3f (±%.3f)", overall_acc, np.std(fold_accs))
    logger.info("\n%s", classification_report(y_true, y_pred, target_names=class_names))

    return {
        "accuracy": overall_acc,
        "fold_accuracies": fold_accs,
        "per_class": report,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }


def aggregate_clip_predictions(clip_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate clip-level predictions to video-level by majority vote.

    Args:
        clip_df: DataFrame with 'video_id', 'predicted', 'true_label' columns.

    Returns:
        DataFrame with one row per video: video_id, predicted, true_label.
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
