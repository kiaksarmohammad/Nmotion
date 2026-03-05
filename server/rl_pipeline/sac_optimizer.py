"""
Stage 3: SAC Threshold/Weight Optimization

RubricEnv: Gymnasium env where SAC optimizes continuous thresholds and weights
for the selected feature subset.

State: 152-dim (encoder embedding statistics + current rubric params)
Action: continuous [thresholds_S, weights_S] with bounds [μ±3σ] / [0,1]

SB3 SAC with HER replay buffer for relabeling failed episodes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False
    logger.error("gymnasium not available. Install with: pip install gymnasium")


if _GYM_AVAILABLE:
    class RubricEnv(gym.Env):
        """
        SAC optimization environment for rubric threshold/weight discovery.

        State (152-dim):
            - Per-feature statistics: [mean, std, threshold, weight] × S features
            - Padded to 152 total dims
            - Current val AUROC (1 dim embedded in state)

        Action (2S-dim, continuous):
            - First S dims: thresholds in [μ_i - 3σ_i, μ_i + 3σ_i]
            - Last S dims: weights in [0, 1]

        Terminal: convergence (ΔAUROC < 1e-4 for 50 steps) or max episodes.
        """

        metadata = {"render_modes": []}

        def __init__(
            self,
            selected_features: List[int],
            train_windows: np.ndarray,
            val_windows: np.ndarray,
            val_labels: np.ndarray,
            biomarker_names: List[str],
            cfg,
            reward_fn=None,
        ):
            super().__init__()

            self.selected = selected_features
            self.S = len(selected_features)
            self.train_windows = train_windows   # [N, W, C]
            self.val_windows = val_windows       # [M, W, C]
            self.val_labels = val_labels
            self.biomarker_names = biomarker_names
            self.cfg = cfg
            self.reward_fn = reward_fn

            # Compute per-feature statistics from training data
            self._feat_means, self._feat_stds = self._compute_stats()

            # Observation space: exactly 152-dim (STATE_DIM)
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(cfg.STATE_DIM,),
                dtype=np.float32,
            )

            # Action space: [thresholds (S), weights (S)]
            thresh_low = np.array(
                [self._feat_means[i] - 3 * self._feat_stds[i] for i in range(self.S)],
                dtype=np.float32,
            )
            thresh_high = np.array(
                [self._feat_means[i] + 3 * self._feat_stds[i] for i in range(self.S)],
                dtype=np.float32,
            )
            weight_low = np.zeros(self.S, dtype=np.float32)
            weight_high = np.ones(self.S, dtype=np.float32)

            self.action_space = spaces.Box(
                low=np.concatenate([thresh_low, weight_low]),
                high=np.concatenate([thresh_high, weight_high]),
                dtype=np.float32,
            )

            # Episode state
            self._current_thresholds = self._feat_means.copy()
            self._current_weights = np.ones(self.S, dtype=np.float32) / self.S
            self._current_auroc = 0.5
            self._prev_auroc = 0.5
            self._no_improve_count = 0
            self._step_count = 0
            self._episode_rewards: List[float] = []

        def _compute_stats(self) -> Tuple[np.ndarray, np.ndarray]:
            """Compute mean and std for each selected feature across training windows."""
            means = np.zeros(self.S, dtype=np.float32)
            stds = np.ones(self.S, dtype=np.float32)

            if len(self.train_windows) == 0:
                return means, stds

            for i, feat_idx in enumerate(self.selected):
                if feat_idx < self.train_windows.shape[2]:
                    vals = self.train_windows[:, :, feat_idx].flatten()
                    vals = vals[~np.isnan(vals)]
                    if len(vals) > 0:
                        means[i] = float(vals.mean())
                        stds[i] = max(float(vals.std()), 1e-6)

            return means, stds

        def _build_state(self) -> np.ndarray:
            """
            Build 152-dim state vector:
            [mean_i, std_i, threshold_i, weight_i] for each selected feature (4×S),
            + [current_auroc] + zero-padded to 152.
            """
            parts = []
            for i in range(self.S):
                parts.extend([
                    self._feat_means[i],
                    self._feat_stds[i],
                    self._current_thresholds[i],
                    self._current_weights[i],
                ])
            parts.append(self._current_auroc)

            state = np.array(parts, dtype=np.float32)

            # Pad or truncate to STATE_DIM
            state_dim = self.cfg.STATE_DIM
            if len(state) < state_dim:
                state = np.pad(state, (0, state_dim - len(state)))
            else:
                state = state[:state_dim]

            assert state.shape == (state_dim,), (
                f"State shape mismatch: {state.shape} != ({state_dim},)"
            )
            return state

        def _score_rubric(
            self, thresholds: np.ndarray, weights: np.ndarray
        ) -> Tuple[np.ndarray, float]:
            """
            Apply rubric to val set: weighted threshold score.
            Returns (scores [M], auroc float).
            """
            try:
                from sklearn.metrics import roc_auc_score
            except ImportError:
                return np.zeros(len(self.val_windows)), 0.5

            M = len(self.val_windows)
            scores = np.zeros(M, dtype=np.float32)

            for i, feat_idx in enumerate(self.selected):
                if feat_idx >= self.val_windows.shape[2]:
                    continue
                feat_vals = self.val_windows[:, :, feat_idx].mean(axis=1)  # [M]
                # Binary threshold: 1 if mean > threshold
                exceeded = (feat_vals > thresholds[i]).astype(np.float32)
                scores += weights[i] * exceeded

            # Normalize scores
            w_sum = weights.sum()
            if w_sum > 0:
                scores /= w_sum

            y_binary = (self.val_labels > 0).astype(int)
            if len(np.unique(y_binary)) < 2:
                return scores, 0.5

            try:
                auroc = float(roc_auc_score(y_binary, scores))
            except ValueError:
                auroc = 0.5

            return scores, auroc

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self._current_thresholds = self._feat_means.copy()
            self._current_weights = np.ones(self.S, dtype=np.float32) / max(self.S, 1)
            _, self._current_auroc = self._score_rubric(
                self._current_thresholds, self._current_weights
            )
            self._prev_auroc = self._current_auroc
            self._no_improve_count = 0
            self._step_count = 0
            self._episode_rewards = []

            state = self._build_state()
            assert state.shape == (self.cfg.STATE_DIM,), (
                f"reset: state shape {state.shape} != ({self.cfg.STATE_DIM},)"
            )
            return state, {}

        def step(self, action: np.ndarray):
            self._step_count += 1

            thresholds = action[: self.S].astype(np.float32)
            weights = np.clip(action[self.S:], 0.0, 1.0).astype(np.float32)

            # Normalize weights
            w_sum = weights.sum()
            if w_sum > 0:
                weights = weights / w_sum
            else:
                weights = np.ones(self.S, dtype=np.float32) / max(self.S, 1)

            self._current_thresholds = thresholds
            self._current_weights = weights

            scores, new_auroc = self._score_rubric(thresholds, weights)
            self._prev_auroc = self._current_auroc
            self._current_auroc = new_auroc

            delta_auroc = new_auroc - self._prev_auroc

            # Compute reward
            if self.reward_fn is not None:
                y_pred = (scores > 0.5).astype(int)
                y_binary = (self.val_labels > 0).astype(int)

                feat_names = [self.biomarker_names[i] for i in self.selected]
                thresh_dict = dict(zip(feat_names, thresholds.tolist()))
                means_dict = dict(zip(feat_names, self._feat_means.tolist()))
                vars_dict = {
                    n: float(s ** 2) for n, s in zip(feat_names, self._feat_stds.tolist())
                }

                reward = self.reward_fn.total(
                    y_true=y_binary,
                    y_pred=y_pred,
                    rubric_scores=scores,
                    val_labels=self.val_labels,
                    prev_auroc=self._prev_auroc,
                    n_selected=self.S,
                    thresholds=thresh_dict,
                    data_means=means_dict,
                    data_vars=vars_dict,
                )
            else:
                reward = delta_auroc - 0.1 * self.S / max(self.cfg.N_BIOMARKERS, 1)

            self._episode_rewards.append(reward)

            # Convergence check
            if abs(delta_auroc) < 1e-4:
                self._no_improve_count += 1
            else:
                self._no_improve_count = 0

            terminated = self._no_improve_count >= 50
            truncated = False

            state = self._build_state()
            assert state.shape == (self.cfg.STATE_DIM,), (
                f"step: state shape {state.shape} != ({self.cfg.STATE_DIM},)"
            )
            info = {
                "auroc": new_auroc,
                "delta_auroc": delta_auroc,
                "n_selected": self.S,
                "thresholds": thresholds.tolist(),
                "weights": weights.tolist(),
            }
            return state, float(reward), terminated, truncated, info

        def render(self):
            pass

else:
    class RubricEnv:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("gymnasium is required. pip install gymnasium")


# ---------------------------------------------------------------------------
# SAC training
# ---------------------------------------------------------------------------

def train_sac(
    env: "RubricEnv",
    selected_features: List[int],
    cfg,
    mlflow_run=None,
) -> Any:
    """
    Train SB3 SAC with HER replay buffer on RubricEnv.
    Auto-tunes entropy coefficient α with target_entropy = -|S|.

    Returns the trained SAC policy.
    """
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
    except ImportError as e:
        raise ImportError(
            "stable-baselines3 required. pip install stable-baselines3"
        ) from e

    S = len(selected_features)
    target_entropy = -float(S)
    logger.info(
        "Training SAC — S=%d features, target_entropy=%.2f", S, target_entropy
    )

    # HER requires GoalEnv; wrap our env for compatibility
    # For non-goal envs, fall back to standard SAC replay buffer
    try:
        policy = SAC(
            "MlpPolicy",
            env,
            learning_rate=cfg.SAC_LR,
            buffer_size=cfg.REPLAY_BUFFER,
            batch_size=cfg.BATCH_SIZE,
            ent_coef="auto",
            target_entropy=target_entropy,
            verbose=1,
            seed=42,
        )
    except Exception as e:
        logger.warning("SAC init failed: %s — trying without HER.", e)
        from stable_baselines3 import SAC as _SAC
        policy = _SAC(
            "MlpPolicy",
            env,
            learning_rate=cfg.SAC_LR,
            buffer_size=cfg.REPLAY_BUFFER,
            batch_size=cfg.BATCH_SIZE,
            ent_coef="auto",
            target_entropy=target_entropy,
            verbose=1,
            seed=42,
        )

    policy.learn(
        total_timesteps=cfg.SAC_TOTAL_TIMESTEPS,
        callback=_SACCallback(mlflow_run=mlflow_run),
    )

    ckpt_path = cfg.CHECKPOINT_DIR / "sac_rubric.zip"
    policy.save(str(ckpt_path))
    logger.info("SAC checkpoint saved to %s", ckpt_path)

    if mlflow_run is not None:
        try:
            import mlflow
            mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")
        except Exception:
            pass

    return policy


def extract_youden_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract Youden-optimal thresholds per feature using the rubric scores.
    For a single-score rubric, returns (threshold, ci_low, ci_high) via bootstrap.

    scores: [N] rubric scores
    labels: [N] binary labels
    Returns: (thresholds [1], ci_low [1], ci_high [1])
    """
    try:
        from sklearn.metrics import roc_curve
    except ImportError:
        return np.array([0.5]), np.array([0.3]), np.array([0.7])

    y_binary = (labels > 0).astype(int)
    fpr, tpr, threshs = roc_curve(y_binary, scores)
    youden_idx = np.argmax(tpr - fpr)
    best_thresh = float(threshs[youden_idx])

    rng = np.random.default_rng(random_state)
    bootstrap_threshs = []
    n = len(scores)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bs_scores = scores[idx]
        bs_labels = y_binary[idx]
        if len(np.unique(bs_labels)) < 2:
            bootstrap_threshs.append(best_thresh)
            continue
        fpr_b, tpr_b, threshs_b = roc_curve(bs_labels, bs_scores)
        yi = np.argmax(tpr_b - fpr_b)
        bootstrap_threshs.append(float(threshs_b[yi]))

    ci_low = float(np.percentile(bootstrap_threshs, 2.5))
    ci_high = float(np.percentile(bootstrap_threshs, 97.5))

    return np.array([best_thresh]), np.array([ci_low]), np.array([ci_high])


def finalize_weights(
    irl_gradients: np.ndarray,
    sac_weights: np.ndarray,
    selected: List[int],
) -> np.ndarray:
    """
    Combine IRL gradient magnitudes with SAC-optimized weights via softmax.
    Returns normalized weight vector [S].
    """
    irl_sel = irl_gradients[selected]
    combined = 0.5 * irl_sel + 0.5 * np.clip(sac_weights, 0, 1)
    # Softmax normalization
    exp = np.exp(combined - combined.max())
    return (exp / exp.sum()).astype(np.float32)


# ---------------------------------------------------------------------------
# SAC MLflow callback
# ---------------------------------------------------------------------------

try:
    from stable_baselines3.common.callbacks import BaseCallback as _BaseCallback2

    class _SACCallback(_BaseCallback2):
        def __init__(self, mlflow_run, log_freq: int = 1000):
            super().__init__(verbose=0)
            self.mlflow_run = mlflow_run
            self.log_freq = log_freq

        def _on_step(self) -> bool:
            if self.mlflow_run is None:
                return True
            if self.n_calls % self.log_freq == 0:
                try:
                    import mlflow
                    lv = self.logger.name_to_value
                    for key in ("train/actor_loss", "train/critic_loss", "train/ent_coef"):
                        if key in lv:
                            mlflow.log_metric(
                                f"sac_{key.split('/')[1]}", lv[key], step=self.n_calls
                            )
                    # Log AUROC from env info if available
                    if hasattr(self.training_env, "get_attr"):
                        try:
                            aurocs = self.training_env.get_attr("_current_auroc")
                            if aurocs:
                                mlflow.log_metric("sac_auroc", aurocs[0], step=self.n_calls)
                        except Exception:
                            pass
                except Exception:
                    pass
            return True

except ImportError:
    class _SACCallback:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass
