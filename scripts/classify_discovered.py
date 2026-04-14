#!/usr/bin/env python3
"""XGBoost classification using discovered features.

Runs three experiments:
1. Top-K features (by Kruskal-Wallis ranking)
2. All significant features (p < 0.05)
3. Original 11 baseline features

Reports per-class F1, confusion matrix, and feature importances.

Usage:
    python scripts/classify_discovered.py \
        --features output/feature_discovery/all_features.csv \
        --ranking output/feature_discovery/ranking.csv \
        --output-dir output/classification
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("classify")

# Original 11 features from pipeline/classify.py
BASELINE_FEATURES = [
    "sample_entropy",
    "spectral_entropy",
    "dfa_alpha",
    "symmetry_mean",  # was symmetry_index in old pipeline
    "peak_frequency",
    "ke_mean",  # was kinetic_energy_mean
    "ke_std",  # was kinetic_energy_std
    "flow_mean",
    "flow_std",
    "flow_skew",
    "flow_kurtosis",
]


def run_cv_experiment(
    df: pd.DataFrame,
    feature_cols: list[str],
    experiment_name: str,
    n_folds: int = 5,
) -> dict:
    """Run stratified grouped CV with XGBoost."""
    # Filter to features that actually exist and have data
    valid_cols = [c for c in feature_cols if c in df.columns]
    if not valid_cols:
        logger.warning("No valid features for experiment '%s'", experiment_name)
        return {}

    # Drop rows with all-NaN features
    X_df = df[valid_cols].copy()
    valid_mask = X_df.notna().any(axis=1)
    df_clean = df[valid_mask].copy()
    X = df_clean[valid_cols].fillna(0).values

    le = LabelEncoder()
    y = le.fit_transform(df_clean["group"])
    # Use video name as group — each video is its own group
    groups = df_clean["video"].values

    # Class weights: inverse frequency
    counts = df_clean["group"].value_counts()
    n_total = len(df_clean)
    n_classes = len(counts)
    class_weight_map = {
        cls: n_total / (n_classes * count)
        for cls, count in counts.items()
    }
    sample_weights = np.array([class_weight_map[g] for g in df_clean["group"]])

    # Adjust n_folds if too few samples in any class
    min_class_count = counts.min()
    actual_folds = min(n_folds, min_class_count)
    if actual_folds < 2:
        logger.warning(
            "Too few samples in smallest class (%d) for CV in '%s'",
            min_class_count, experiment_name,
        )
        actual_folds = 2

    sgkf = StratifiedGroupKFold(
        n_splits=actual_folds, shuffle=True, random_state=42,
    )

    all_preds = np.full(len(df_clean), -1, dtype=int)
    all_proba = np.zeros((len(df_clean), n_classes))
    fold_accs = []
    fold_f1s = []
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
        model.fit(X[train_idx], y[train_idx], sample_weight=sample_weights[train_idx])

        preds = model.predict(X[test_idx])
        proba = model.predict_proba(X[test_idx])
        all_preds[test_idx] = preds
        all_proba[test_idx] = proba

        fold_acc = accuracy_score(y[test_idx], preds)
        fold_f1 = f1_score(y[test_idx], preds, average="weighted")
        fold_accs.append(fold_acc)
        fold_f1s.append(fold_f1)
        importances_sum += model.feature_importances_

        logger.info(
            "  [%s] Fold %d: acc=%.3f  f1=%.3f",
            experiment_name, fold_idx + 1, fold_acc, fold_f1,
        )

    # Overall metrics
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

    logger.info("=" * 60)
    logger.info("[%s] RESULTS (%d features, %d folds)", experiment_name, len(valid_cols), actual_folds)
    logger.info("  Accuracy: %.3f (±%.3f)", overall_acc, np.std(fold_accs))
    logger.info("  Weighted F1: %.3f (±%.3f)", overall_f1, np.std(fold_f1s))
    logger.info("\n%s", classification_report(y_true, y_pred, target_names=class_names))
    logger.info("Confusion matrix:\n%s", cm)
    logger.info("=" * 60)

    # Normalized feature importances
    avg_importance = importances_sum / actual_folds
    importance_df = pd.DataFrame({
        "feature": valid_cols,
        "importance": avg_importance,
    }).sort_values("importance", ascending=False)

    return {
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
        "feature_importances": importance_df,
        "class_weights": class_weight_map,
    }


def plot_comparison(results: list[dict], output_dir: Path) -> None:
    """Bar chart comparing experiments."""
    names = [r["name"] for r in results]
    accs = [r["accuracy"] for r in results]
    f1s = [r["weighted_f1"] for r in results]
    acc_stds = [r["accuracy_std"] for r in results]
    f1_stds = [r["f1_std"] for r in results]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, accs, width, yerr=acc_stds, label="Accuracy",
                   color="#3498db", capsize=5, alpha=0.85)
    bars2 = ax.bar(x + width / 2, f1s, width, yerr=f1_stds, label="Weighted F1",
                   color="#e74c3c", capsize=5, alpha=0.85)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("XGBoost Classification — Feature Set Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "experiment_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "experiment_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(results: list[dict], output_dir: Path) -> None:
    """Side-by-side confusion matrices for each experiment."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        cm = np.array(r["confusion_matrix"])
        class_names = r["class_names"]

        # Normalize by row (true label)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                text = f"{cm[i, j]}\n({cm_norm[i, j]:.0%})"
                color = "white" if cm_norm[i, j] > 0.6 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=10, color=color)

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=10)
        ax.set_yticklabels(class_names, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.set_title(f"{r['name']}\nAcc={r['accuracy']:.2f}  F1={r['weighted_f1']:.2f}",
                      fontsize=11, fontweight="bold")

    fig.suptitle("Confusion Matrices — XGBoost with Grouped CV", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrices.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "confusion_matrices.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_feature_importances(results: list[dict], output_dir: Path) -> None:
    """Feature importance plot for the best experiment."""
    # Pick the experiment with highest F1
    best = max(results, key=lambda r: r["weighted_f1"])
    imp = best["feature_importances"].head(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(imp))[::-1]
    ax.barh(y_pos, imp["importance"].values, color="#2ecc71", edgecolor="white", height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(imp["feature"].values, fontsize=9)
    ax.set_xlabel("Feature Importance (avg gain)", fontsize=11)
    ax.set_title(
        f"Top 20 Feature Importances — {best['name']}",
        fontsize=13, fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "feature_importances.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "feature_importances.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="XGBoost classification with discovered features")
    parser.add_argument("--features", type=Path, required=True, help="all_features.csv from discovery")
    parser.add_argument("--ranking", type=Path, required=True, help="ranking.csv from discovery")
    parser.add_argument("--output-dir", type=Path, default=Path("output/classification"))
    parser.add_argument("--top-k", type=int, default=15, help="Number of top features for top-K experiment")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    ranking = pd.read_csv(args.ranking)

    logger.info("Loaded %d videos, %d ranked features", len(df), len(ranking))
    for g in sorted(df["group"].unique()):
        logger.info("  %s: %d videos", g, (df["group"] == g).sum())

    # Define experiments
    sig_features = ranking[ranking["kw_pvalue"] < 0.05]["feature"].tolist()
    top_k_features = ranking["feature"].head(args.top_k).tolist()
    baseline_present = [f for f in BASELINE_FEATURES if f in df.columns]

    experiments = [
        (f"Top-{args.top_k} Discovered", top_k_features),
        (f"All Significant (n={len(sig_features)})", sig_features),
        (f"Baseline (n={len(baseline_present)})", baseline_present),
    ]

    results = []
    for name, feat_cols in experiments:
        logger.info("\n>>> Experiment: %s (%d features)", name, len(feat_cols))
        r = run_cv_experiment(df, feat_cols, name)
        if r:
            results.append(r)

    if not results:
        logger.error("No experiments completed!")
        return

    # Generate plots
    logger.info("\nGenerating plots...")
    plot_comparison(results, output_dir)
    plot_confusion_matrices(results, output_dir)
    plot_feature_importances(results, output_dir)

    # Save summary
    summary_rows = []
    for r in results:
        row = {
            "experiment": r["name"],
            "n_features": r["n_features"],
            "accuracy": r["accuracy"],
            "accuracy_std": r["accuracy_std"],
            "weighted_f1": r["weighted_f1"],
            "f1_std": r["f1_std"],
        }
        # Per-class F1
        for cls in r["class_names"]:
            if cls in r["per_class"]:
                row[f"f1_{cls}"] = r["per_class"][cls]["f1-score"]
                row[f"precision_{cls}"] = r["per_class"][cls]["precision"]
                row[f"recall_{cls}"] = r["per_class"][cls]["recall"]
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "classification_summary.csv", index=False)

    # Save detailed results as JSON
    json_results = []
    for r in results:
        r_copy = {k: v for k, v in r.items() if k != "feature_importances"}
        json_results.append(r_copy)
    with open(output_dir / "classification_results.json", "w") as f:
        json.dump(json_results, f, indent=2, default=str)

    # Save feature importances
    for r in results:
        safe_name = r["name"].replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        r["feature_importances"].to_csv(
            output_dir / f"importances_{safe_name}.csv", index=False,
        )

    logger.info("\n" + "=" * 60)
    logger.info("CLASSIFICATION COMPLETE")
    logger.info("=" * 60)
    for r in results:
        logger.info(
            "  %-40s  acc=%.3f  f1=%.3f",
            r["name"], r["accuracy"], r["weighted_f1"],
        )
    logger.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
