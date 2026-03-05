"""
Neuromotion-AI — Three-Stage Offline RL Pipeline
CLI entry point.

Usage:
    python server/rl_pipeline/train.py [--data-dir data/] [--output-dir output/] [--epochs 200]

Stages:
    1. Contrastive transformer encoder pre-training (NT-Xent)
    2A. Behavioral cloning + MaxEnt IRL
    2B. MARL feature selection (PettingZoo + SB3 PPO)
    3.  SAC threshold/weight optimization (SB3 + HER)
    V.  Validation (cross-OPE, null test, temporal test, plausibility, sensitivity)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Add server/ to path so relative imports work when run as a script
_SERVER_DIR = Path(__file__).parent.parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from rl_pipeline.config import PipelineConfig
from rl_pipeline.data_loader import (
    WeightedEpisodeSampler,
    apply_smote_ts,
    fit_scaler,
    load_all_recordings,
    recordings_to_windows,
    scale_recording,
    stratified_split,
)
from rl_pipeline.encoder import encode_windows, pretrain_encoder
from rl_pipeline.irl import (
    build_expert_sequences,
    compute_gradient_magnitudes,
    fit_behavioral_cloning,
    get_marl_priors,
    maxent_irl_train,
)
from rl_pipeline.marl_env import FeatureSelectionEnv, get_selected_features, train_marl_agents
from rl_pipeline.reward import CompositeReward
from rl_pipeline.rubric import (
    NeonatalRubric,
    compute_shap_attribution,
    export_rubric_report,
    fit_supervised_baseline,
)
from rl_pipeline.sac_optimizer import (
    RubricEnv,
    extract_youden_thresholds,
    finalize_weights,
    train_sac,
)
from rl_pipeline.validation import (
    RubricFailureError,
    biomarker_plausibility_audit,
    check_hard_failures,
    compute_test_metrics,
    cross_ope_matrix,
    null_shuffle_test,
    score_rubric_on_windows,
    sensitivity_analysis,
    temporal_shuffle_test,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Neuromotion-AI: Three-Stage Offline RL Pipeline"
    )
    parser.add_argument(
        "--data-dir", default="data/", help="Directory containing neuromotion_*.csv files"
    )
    parser.add_argument(
        "--output-dir", default="output/", help="Output directory for reports and checkpoints"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Encoder pre-training epochs (overrides config default)"
    )
    parser.add_argument(
        "--skip-marl", action="store_true",
        help="Skip MARL feature selection and use all biomarkers"
    )
    parser.add_argument(
        "--skip-sac", action="store_true",
        help="Skip SAC optimization (use IRL-derived thresholds only)"
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device (cuda / cpu). Auto-detected if not specified."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_load_data(cfg: PipelineConfig):
    """Load, preprocess, and split recordings."""
    logger.info("=" * 60)
    logger.info("DATA LOADING")
    logger.info("=" * 60)

    recordings = load_all_recordings(cfg.DATA_DIR, cfg.BIOMARKER_NAMES)
    if not recordings:
        logger.warning(
            "No data found in %s. Generating synthetic demo data for pipeline smoke-test.",
            cfg.DATA_DIR,
        )
        recordings = _generate_demo_data(cfg)

    train_recs, val_recs, test_recs = stratified_split(
        recordings,
        train_frac=cfg.TRAIN_FRAC,
        val_frac=cfg.VAL_FRAC,
    )

    # Fit scaler on train only
    scaler = fit_scaler(train_recs, cfg.BIOMARKER_NAMES)

    train_recs = [scale_recording(r, scaler, cfg.BIOMARKER_NAMES) for r in train_recs]
    val_recs = [scale_recording(r, scaler, cfg.BIOMARKER_NAMES) for r in val_recs]
    test_recs = [scale_recording(r, scaler, cfg.BIOMARKER_NAMES) for r in test_recs]

    # Window extraction
    train_w, train_l = recordings_to_windows(
        train_recs, cfg.BIOMARKER_NAMES, cfg.WINDOW_SIZE, cfg.STRIDE
    )
    val_w, val_l = recordings_to_windows(
        val_recs, cfg.BIOMARKER_NAMES, cfg.WINDOW_SIZE, cfg.STRIDE
    )
    test_w, test_l = recordings_to_windows(
        test_recs, cfg.BIOMARKER_NAMES, cfg.WINDOW_SIZE, cfg.STRIDE
    )

    # SMOTE-TS on training pathological windows
    patho_mask = train_l > 0
    if patho_mask.sum() > 0:
        patho_w = apply_smote_ts(
            train_w[patho_mask], target_ratio=cfg.SMOTE_TARGET_RATIO
        )
        normal_w = train_w[~patho_mask]
        normal_l = train_l[~patho_mask]
        synthetic_l = np.ones(len(patho_w), dtype=np.int64)
        train_w = np.concatenate([normal_w, patho_w], axis=0)
        train_l = np.concatenate([normal_l, synthetic_l], axis=0)
        logger.info(
            "After SMOTE-TS: %d training windows (%d patho, %d normal)",
            len(train_w), patho_mask.sum(), (~patho_mask).sum(),
        )

    logger.info(
        "Windows — train=%s  val=%s  test=%s",
        train_w.shape, val_w.shape, test_w.shape,
    )
    return train_w, train_l, val_w, val_l, test_w, test_l, scaler


def stage1_encoder(train_w, train_l, cfg, device, run):
    """Stage 1: Pre-train contrastive encoder."""
    logger.info("=" * 60)
    logger.info("STAGE 1: CONTRASTIVE ENCODER PRE-TRAINING")
    logger.info("=" * 60)

    encoder = pretrain_encoder(
        train_windows=train_w,
        train_labels=train_l,
        cfg=cfg,
        mlflow_run=run,
        device=device,
    )

    # Encode all splits
    train_emb = encode_windows(encoder, train_w, device=device)
    logger.info("Train embeddings: %s", train_emb.shape)
    return encoder, train_emb


def stage2a_irl(encoder, train_w, train_l, train_emb, cfg, device, run):
    """Stage 2A: Behavioral Cloning + MaxEnt IRL."""
    logger.info("=" * 60)
    logger.info("STAGE 2A: BEHAVIORAL CLONING + IRL")
    logger.info("=" * 60)

    expert_states, expert_actions = build_expert_sequences(train_emb, train_l, cfg)

    bc_model = fit_behavioral_cloning(
        expert_states=expert_states,
        expert_actions=expert_actions,
        cfg=cfg,
        mlflow_run=run,
        device=device,
    )

    # All embeddings (not just pathological) for IRL partition function
    expert_emb = train_emb[train_l > 0]
    if len(expert_emb) == 0:
        logger.warning("No pathological samples for IRL — using all embeddings as expert.")
        expert_emb = train_emb

    reward_model = maxent_irl_train(
        expert_embeddings=expert_emb,
        all_embeddings=train_emb,
        cfg=cfg,
        mlflow_run=run,
        device=device,
    )

    grad_mags = compute_gradient_magnitudes(reward_model, train_emb, device=device)
    marl_priors = get_marl_priors(grad_mags, cfg.BIOMARKER_NAMES)

    logger.info("IRL gradient magnitudes per biomarker:")
    for i, (name, mag) in enumerate(zip(cfg.BIOMARKER_NAMES, grad_mags)):
        logger.info("  [%d] %-20s %.4f", i, name, mag)

    try:
        import mlflow
        mlflow.log_dict(
            {cfg.BIOMARKER_NAMES[i]: float(grad_mags[i]) for i in range(len(grad_mags))},
            "irl_gradient_magnitudes.json",
        )
    except Exception:
        pass

    return grad_mags, marl_priors


def stage2b_marl(val_w, val_l, grad_mags, marl_priors, cfg, run, skip_marl: bool):
    """Stage 2B: MARL feature selection."""
    logger.info("=" * 60)
    logger.info("STAGE 2B: MARL FEATURE SELECTION")
    logger.info("=" * 60)

    if skip_marl:
        logger.info("--skip-marl: using all %d biomarkers.", cfg.N_BIOMARKERS)
        selected = list(range(cfg.N_BIOMARKERS))
        return selected

    env = FeatureSelectionEnv(
        biomarker_names=cfg.BIOMARKER_NAMES,
        irl_gradients=grad_mags,
        val_windows=val_w,
        val_labels=val_l,
        cfg=cfg,
    )

    policies = train_marl_agents(env, marl_priors, cfg, mlflow_run=run)
    selected = get_selected_features(policies, env, cfg)

    try:
        import mlflow
        mlflow.log_dict(
            {"selected_indices": selected,
             "selected_names": [cfg.BIOMARKER_NAMES[i] for i in selected]},
            "marl_selected_features.json",
        )
    except Exception:
        pass

    return selected


def stage3_sac(
    train_w, train_l, val_w, val_l,
    selected, grad_mags, cfg, run, skip_sac: bool,
):
    """Stage 3: SAC threshold/weight optimization."""
    logger.info("=" * 60)
    logger.info("STAGE 3: SAC THRESHOLD/WEIGHT OPTIMIZATION")
    logger.info("=" * 60)

    reward_fn = CompositeReward(
        cfg=cfg,
        biomarker_names=cfg.BIOMARKER_NAMES,
        clinical_directions=cfg.CLINICAL_DIRECTIONS,
    )

    if skip_sac:
        logger.info("--skip-sac: deriving thresholds from training data statistics.")
        thresholds = {}
        weights = {}
        for i in selected:
            name = cfg.BIOMARKER_NAMES[i]
            feat_vals = train_w[:, :, i].flatten()
            feat_vals = feat_vals[~np.isnan(feat_vals)]
            thresholds[name] = float(np.median(feat_vals))
            weights[name] = float(grad_mags[i])

        # Normalize weights
        w_arr = np.array(list(weights.values()))
        w_arr = np.exp(w_arr - w_arr.max())
        w_arr /= w_arr.sum()
        weights = dict(zip(weights.keys(), w_arr.tolist()))
        return thresholds, weights, None

    env = RubricEnv(
        selected_features=selected,
        train_windows=train_w,
        val_windows=val_w,
        val_labels=val_l,
        biomarker_names=cfg.BIOMARKER_NAMES,
        cfg=cfg,
        reward_fn=reward_fn,
    )

    sac_policy = train_sac(env, selected, cfg, mlflow_run=run)

    # Extract final thresholds/weights from SAC policy
    obs, _ = env.reset()
    action, _ = sac_policy.predict(obs, deterministic=True)
    S = len(selected)
    sac_thresholds_raw = action[:S]
    sac_weights_raw = np.clip(action[S:], 0, 1)

    final_weights = finalize_weights(grad_mags, sac_weights_raw, selected)
    thresholds = {
        cfg.BIOMARKER_NAMES[i]: float(sac_thresholds_raw[j])
        for j, i in enumerate(selected)
    }
    weights = {
        cfg.BIOMARKER_NAMES[i]: float(final_weights[j])
        for j, i in enumerate(selected)
    }

    return thresholds, weights, sac_policy


def stage_validate(
    rubric, train_w, train_l, val_w, val_l, test_w, test_l, cfg, run
):
    """Run all validation checks."""
    logger.info("=" * 60)
    logger.info("VALIDATION")
    logger.info("=" * 60)

    # Base test metrics
    test_metrics = compute_test_metrics(
        rubric, test_w, test_l, cfg.BIOMARKER_NAMES, mlflow_run=run
    )

    # Cross-OPE matrix
    try:
        ope_df = cross_ope_matrix(
            rubric=rubric,
            val_windows=val_w,
            val_labels=val_l,
            biomarker_names=cfg.BIOMARKER_NAMES,
            ope_variants=cfg.OPE_VARIANTS,
            reward_cls=CompositeReward,
            cfg=cfg,
            mlflow_run=run,
        )
    except RubricFailureError as e:
        logger.error("Cross-OPE failure: %s", e)
        raise

    # Null shuffle
    try:
        null_shuffle_test(
            rubric=rubric,
            windows=val_w,
            labels=val_l,
            biomarker_names=cfg.BIOMARKER_NAMES,
            n_shuffles=cfg.NULL_SHUFFLE_N,
            max_null_auroc=cfg.NULL_MAX_AUROC,
            mlflow_run=run,
        )
    except RubricFailureError as e:
        logger.error("Null shuffle failure: %s", e)
        raise

    # Temporal shuffle
    base_auroc = test_metrics.get("auroc", 0.5)
    temporal_shuffle_test(
        rubric=rubric,
        windows=test_w,
        labels=test_l,
        biomarker_names=cfg.BIOMARKER_NAMES,
        base_auroc=base_auroc,
        mlflow_run=run,
    )

    # Plausibility audit
    feat_means = {
        cfg.BIOMARKER_NAMES[i]: float(train_w[:, :, i].mean())
        for i in range(cfg.N_BIOMARKERS)
    }
    try:
        biomarker_plausibility_audit(
            thresholds=rubric.thresholds,
            data_means=feat_means,
            directions=cfg.CLINICAL_DIRECTIONS,
            mlflow_run=run,
        )
    except RubricFailureError as e:
        logger.error("Plausibility audit failure: %s", e)
        raise

    # Sensitivity analysis
    robustness = sensitivity_analysis(
        rubric=rubric,
        test_windows=test_w,
        test_labels=test_l,
        biomarker_names=cfg.BIOMARKER_NAMES,
        target_robustness=cfg.SENSITIVITY_TARGET,
        mlflow_run=run,
    )

    validation_results = {
        **test_metrics,
        "robustness": robustness,
    }

    # Hard failure check
    check_hard_failures(validation_results, mlflow_run=run)

    return validation_results


# ---------------------------------------------------------------------------
# Demo data generator (for smoke-testing with no real CSVs)
# ---------------------------------------------------------------------------

def _generate_demo_data(cfg: PipelineConfig):
    """Generate synthetic recordings for pipeline smoke-testing."""
    import pandas as pd
    from rl_pipeline.data_loader import Recording

    logger.warning(
        "Generating SYNTHETIC demo data — results will be meaningless for clinical use."
    )
    rng = np.random.default_rng(42)
    recordings = []

    for cat, label, n_subj in [("n1", 0, 5), ("spasms4", 1, 3)]:
        for subj in range(n_subj):
            T = rng.integers(60, 120)
            data = rng.standard_normal((T, cfg.N_BIOMARKERS)).astype(np.float32)
            if label == 1:
                data += 0.5  # shift pathological slightly
            df = pd.DataFrame(data, columns=cfg.BIOMARKER_NAMES)
            df["timestamp"] = np.arange(T)
            recordings.append(
                Recording(
                    df=df,
                    category=cat,
                    label=label,
                    subject_id=f"{cat}_subj{subj:02d}",
                    fname=f"neuromotion_{cat}_subj{subj:02d}_demo.csv",
                )
            )

    return recordings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = PipelineConfig.from_args(args)
    device = args.device

    try:
        import mlflow
        mlflow.set_tracking_uri(cfg.MLFLOW_URI)
        mlflow.set_experiment("neuromotion_rl")
        run_ctx = mlflow.start_run()
        run = run_ctx.__enter__()
        mlflow.log_params({
            "data_dir": str(cfg.DATA_DIR),
            "output_dir": str(cfg.OUTPUT_DIR),
            "encoder_epochs": cfg.ENCODER_EPOCHS,
            "state_dim": cfg.STATE_DIM,
            "n_biomarkers": cfg.N_BIOMARKERS,
            "window_size": cfg.WINDOW_SIZE,
            "lambda1": cfg.LAMBDA1,
            "lambda2": cfg.LAMBDA2,
            "lambda3": cfg.LAMBDA3,
        })
    except ImportError:
        logger.warning("mlflow not installed — running without experiment tracking.")
        run = None
        run_ctx = None

    try:
        # Stage 0: Data
        train_w, train_l, val_w, val_l, test_w, test_l, scaler = stage_load_data(cfg)

        # Stage 1: Encoder
        encoder, train_emb = stage1_encoder(train_w, train_l, cfg, device, run)

        # Stage 2A: IRL
        grad_mags, marl_priors = stage2a_irl(
            encoder, train_w, train_l, train_emb, cfg, device, run
        )

        # Stage 2B: MARL
        selected = stage2b_marl(
            val_w, val_l, grad_mags, marl_priors, cfg, run,
            skip_marl=getattr(args, "skip_marl", False),
        )

        # Stage 3: SAC
        thresholds, weights, sac_policy = stage3_sac(
            train_w, train_l, val_w, val_l,
            selected, grad_mags, cfg, run,
            skip_sac=getattr(args, "skip_sac", False),
        )

        # Compute per-feature CI from val set (use partial rubric for scoring)
        _partial_rubric = NeonatalRubric(
            selected_features=[cfg.BIOMARKER_NAMES[i] for i in selected],
            thresholds=thresholds,
            weights=weights,
        )
        val_scores = score_rubric_on_windows(
            _partial_rubric,
            val_w,
            cfg.BIOMARKER_NAMES,
        )
        thresh_arr, ci_low, ci_high = extract_youden_thresholds(val_scores, val_l)

        # Build threshold CI dict (single global threshold → apply to all features)
        threshold_ci = {
            cfg.BIOMARKER_NAMES[i]: (float(ci_low[0]), float(ci_high[0]))
            for i in selected
        }

        # Build rubric
        feat_means_train = {
            cfg.BIOMARKER_NAMES[i]: float(train_w[:, :, i].mean())
            for i in range(cfg.N_BIOMARKERS)
        }
        rubric = NeonatalRubric(
            selected_features=[cfg.BIOMARKER_NAMES[i] for i in selected],
            thresholds=thresholds,
            weights=weights,
            threshold_ci=threshold_ci,
            clinical_directions={
                cfg.BIOMARKER_NAMES[i]: cfg.CLINICAL_DIRECTIONS.get(
                    cfg.BIOMARKER_NAMES[i], "?"
                )
                for i in selected
            },
            metadata={
                "pipeline_version": "1.0.0",
                "n_train_windows": len(train_w),
                "n_val_windows": len(val_w),
                "n_test_windows": len(test_w),
            },
        )

        # Validation
        validation_results = stage_validate(
            rubric, train_w, train_l, val_w, val_l, test_w, test_l, cfg, run
        )
        rubric.auroc = validation_results.get("auroc", 0.0)

        # SHAP attribution
        shap_vals = compute_shap_attribution(
            rubric=rubric,
            windows=test_w,
            labels=test_l,
            biomarker_names=cfg.BIOMARKER_NAMES,
            cfg=cfg,
            output_dir=cfg.OUTPUT_DIR,
        )
        if shap_vals is not None and run is not None:
            try:
                import mlflow
                shap_chart = cfg.OUTPUT_DIR / "shap_attribution.png"
                if shap_chart.exists():
                    mlflow.log_artifact(str(shap_chart), artifact_path="reports")
            except Exception:
                pass

        # Export report
        report_path = cfg.OUTPUT_DIR / "rubric_report.json"
        export_rubric_report(
            rubric=rubric,
            validation_results=validation_results,
            shap_values=shap_vals,
            path=report_path,
            mlflow_run=run,
        )

        logger.info("\n%s", rubric)
        logger.info("\nReport saved to: %s", report_path)
        logger.info("Run complete. Final AUROC=%.4f", rubric.auroc)

    except RubricFailureError as e:
        logger.error("RUBRIC FAILED VALIDATION: %s", e)
        logger.info("Falling back to supervised Random Forest baseline...")

        # Variables may be unbound if failure happened early in the pipeline
        _locals = locals()
        if all(k in _locals for k in ("train_w", "train_l", "test_w", "test_l")):
            baseline_auroc, _ = fit_supervised_baseline(
                _locals["train_w"], _locals["train_l"],
                _locals["test_w"], _locals["test_l"],
                cfg.BIOMARKER_NAMES, cfg, run,
            )
            logger.info("Baseline RF AUROC: %.4f", baseline_auroc)

        sys.exit(1)

    finally:
        if run_ctx is not None:
            try:
                run_ctx.__exit__(None, None, None)
            except Exception:
                pass


if __name__ == "__main__":
    main()
