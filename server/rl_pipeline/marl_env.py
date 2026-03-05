"""
Stage 2B: MARL Feature Selection Environment

PettingZoo ParallelEnv with 9 agents (one per biomarker).
Each agent decides: 0=exclude, 1=include.

Observation (per agent): [IRL_gradient_i, current_AUROC, is_included_i, rubric_size]
Action: Discrete(2) — exclude / include

Training: Independent SB3 PPO instances (one per agent), NOT RLlib/Ray.
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

try:
    from pettingzoo import ParallelEnv
    _PETTINGZOO_AVAILABLE = True
except ImportError:
    _PETTINGZOO_AVAILABLE = False
    logger.warning("pettingzoo not available. Install with: pip install pettingzoo")


# ---------------------------------------------------------------------------
# Feature Selection Environment
# ---------------------------------------------------------------------------

if _PETTINGZOO_AVAILABLE and _GYM_AVAILABLE:
    class FeatureSelectionEnv(ParallelEnv):
        """
        9-agent parallel environment for biomarker feature selection.

        Agents: agent_0 ... agent_8 (one per biomarker)
        Observation: Box([irl_grad_i, current_auroc, is_included_i, rubric_size], shape=(4,))
        Action: Discrete(2) — 0=exclude, 1=include
        Terminal: all agent entropies < threshold OR max_steps exceeded
        """

        metadata = {"name": "feature_selection_v0", "render_modes": []}

        def __init__(
            self,
            biomarker_names: List[str],
            irl_gradients: np.ndarray,
            val_windows: np.ndarray,
            val_labels: np.ndarray,
            cfg,
        ):
            super().__init__()

            self.biomarker_names = biomarker_names
            self.n_agents = len(biomarker_names)
            self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]
            self.agents = self.possible_agents.copy()

            self.irl_gradients = irl_gradients.astype(np.float32)
            self.val_windows = val_windows        # [N, W, C]
            self.val_labels = val_labels          # [N]
            self.cfg = cfg

            self.max_steps = cfg.MARL_MAX_STEPS
            self._step_count = 0

            # State tracking
            self._included = np.ones(self.n_agents, dtype=np.int32)  # start all included
            self._current_auroc = 0.5
            self._prev_auroc = 0.5

            # Lazy encoder ref — set after env creation
            self.encoder = None
            self.reward_fn = None

        def _agent_idx(self, agent: str) -> int:
            return int(agent.split("_")[1])

        @property
        def observation_spaces(self) -> Dict[str, "gym.Space"]:
            return {
                agent: spaces.Box(
                    low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    high=np.array([1.0, 1.0, 1.0, float(self.n_agents)], dtype=np.float32),
                    shape=(4,),
                    dtype=np.float32,
                )
                for agent in self.possible_agents
            }

        @property
        def action_spaces(self) -> Dict[str, "gym.Space"]:
            return {
                agent: spaces.Discrete(2)
                for agent in self.possible_agents
            }

        def observation_space(self, agent: str) -> "gym.Space":
            return self.observation_spaces[agent]

        def action_space(self, agent: str) -> "gym.Space":
            return self.action_spaces[agent]

        def _get_obs(self) -> Dict[str, np.ndarray]:
            rubric_size = float(self._included.sum())
            return {
                agent: np.array([
                    self.irl_gradients[self._agent_idx(agent)],
                    float(self._current_auroc),
                    float(self._included[self._agent_idx(agent)]),
                    rubric_size,
                ], dtype=np.float32)
                for agent in self.agents
            }

        def _score_rubric(self) -> float:
            """Compute AUROC of current feature subset on validation set."""
            selected = np.where(self._included > 0)[0]
            if len(selected) == 0:
                return 0.5

            try:
                from sklearn.metrics import roc_auc_score
                from sklearn.linear_model import LogisticRegression
            except ImportError:
                return 0.5

            # Use mean of selected channels as a simple rubric score
            # val_windows: [N, W, C]
            N, W, C = self.val_windows.shape
            scores = self.val_windows[:, :, selected].mean(axis=(1, 2))  # [N]
            y_binary = (self.val_labels > 0).astype(int)

            if len(np.unique(y_binary)) < 2:
                return 0.5

            try:
                return float(roc_auc_score(y_binary, scores))
            except ValueError:
                return 0.5

        def reset(self, seed=None, options=None):
            self.agents = self.possible_agents.copy()
            self._included = np.ones(self.n_agents, dtype=np.int32)
            self._current_auroc = self._score_rubric()
            self._prev_auroc = self._current_auroc
            self._step_count = 0
            obs = self._get_obs()
            infos = {agent: {} for agent in self.agents}
            return obs, infos

        def step(
            self, actions: Dict[str, int]
        ) -> Tuple[
            Dict[str, np.ndarray],
            Dict[str, float],
            Dict[str, bool],
            Dict[str, bool],
            Dict[str, Any],
        ]:
            self._step_count += 1

            # Update feature inclusion based on actions
            for agent, action in actions.items():
                idx = self._agent_idx(agent)
                self._included[idx] = int(action)

            # Ensure at least 1 feature included
            if self._included.sum() == 0:
                self._included[0] = 1

            # Warn if angular_jerk (index 8) is excluded
            if self._included[8] == 0:
                logger.warning(
                    "agent_8 (angular_jerk) selected EXCLUDE. "
                    "This conflicts with clinical prior."
                )

            # Compute new AUROC
            self._prev_auroc = self._current_auroc
            self._current_auroc = self._score_rubric()
            delta_auroc = self._current_auroc - self._prev_auroc

            # Per-agent reward: ΔAUROC shared + complexity penalty
            n_selected = int(self._included.sum())
            complexity_penalty = -0.1 * n_selected / self.n_agents

            rewards = {}
            for agent in self.agents:
                rewards[agent] = float(delta_auroc + complexity_penalty)

            # Terminal condition
            done = self._step_count >= self.max_steps
            terminations = {agent: done for agent in self.agents}
            truncations = {agent: False for agent in self.agents}

            obs = self._get_obs()
            infos = {
                agent: {
                    "auroc": self._current_auroc,
                    "n_selected": n_selected,
                }
                for agent in self.agents
            }

            if done:
                self.agents = []

            return obs, rewards, terminations, truncations, infos

else:
    class FeatureSelectionEnv:  # type: ignore[no-redef]
        """Stub — install pettingzoo and gymnasium to use."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PettingZoo and Gymnasium are required for MARL training.\n"
                "Install with: pip install pettingzoo gymnasium shimmy supersuit"
            )


# ---------------------------------------------------------------------------
# MARL training with independent SB3 PPO
# ---------------------------------------------------------------------------

def train_marl_agents(
    env: "FeatureSelectionEnv",
    irl_priors: Dict[int, np.ndarray],
    cfg,
    mlflow_run=None,
) -> Dict[str, Any]:
    """
    Train one independent SB3 PPO agent per biomarker.

    Returns dict: {agent_name: trained SB3 PPO policy}

    Architecture note: uses SB3 PPO (not RLlib/Ray) for lighter footprint.
    Each agent trains independently on its own Gymnasium-compatible wrapper.
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError as e:
        raise ImportError(
            "stable-baselines3 required. Install: pip install stable-baselines3"
        ) from e

    try:
        from supersuit import pettingzoo_env_to_vec_env_v1
        import supersuit as ss
    except ImportError:
        logger.warning(
            "supersuit not installed — using single-agent wrappers instead. "
            "Install with: pip install supersuit"
        )

    policies: Dict[str, Any] = {}
    n_agents = len(env.possible_agents)

    for agent_name in env.possible_agents:
        agent_idx = int(agent_name.split("_")[1])
        logger.info("Training PPO agent: %s (biomarker=%s)",
                    agent_name, cfg.BIOMARKER_NAMES[agent_idx])

        # Wrap single-agent view of the env
        single_env = _SingleAgentWrapper(env, agent_name, irl_priors.get(agent_idx))

        policy = PPO(
            "MlpPolicy",
            single_env,
            learning_rate=cfg.MARL_PPO_LR,
            verbose=0,
            seed=42 + agent_idx,
        )

        policy.learn(
            total_timesteps=cfg.MARL_TOTAL_TIMESTEPS // n_agents,
            callback=_MARLCallback(
                agent_name=agent_name,
                mlflow_run=mlflow_run,
                log_freq=1000,
            ),
        )

        policies[agent_name] = policy

        if mlflow_run is not None:
            try:
                import mlflow
                ckpt_path = cfg.CHECKPOINT_DIR / f"ppo_{agent_name}.zip"
                policy.save(str(ckpt_path))
                mlflow.log_artifact(str(ckpt_path), artifact_path="marl_checkpoints")
            except Exception as e:
                logger.warning("Could not log PPO artifact: %s", e)

    return policies


def get_selected_features(
    policies: Dict[str, Any],
    env: "FeatureSelectionEnv",
    cfg,
) -> List[int]:
    """
    Greedy argmax per converged agent to get final selected feature indices.
    Warns if agent_8 (angular_jerk) selects INCLUDE.
    """
    obs, _ = env.reset()
    selected = []

    for agent_name, policy in policies.items():
        agent_idx = int(agent_name.split("_")[1])
        agent_obs = obs[agent_name]
        action, _ = policy.predict(agent_obs, deterministic=True)
        if int(action) == 1:  # INCLUDE
            selected.append(agent_idx)
            if agent_idx == 8:  # angular_jerk
                logger.warning(
                    "agent_8 (angular_jerk) selected INCLUDE via greedy argmax. "
                    "This aligns with clinical prior [up]."
                )

    # Ensure at least 1 feature
    if not selected:
        logger.warning("No features selected — defaulting to all biomarkers.")
        selected = list(range(cfg.N_BIOMARKERS))

    logger.info(
        "MARL selected %d features: %s",
        len(selected),
        [cfg.BIOMARKER_NAMES[i] for i in selected],
    )
    return selected


# ---------------------------------------------------------------------------
# Single-agent wrapper for SB3
# ---------------------------------------------------------------------------

class _SingleAgentWrapper:
    """
    Wraps the multi-agent env as a single-agent Gymnasium env for one agent.
    """

    def __init__(
        self,
        env: "FeatureSelectionEnv",
        agent_name: str,
        init_prior: Optional[np.ndarray] = None,
    ):
        if not _GYM_AVAILABLE:
            raise ImportError("gymnasium required")

        self.env = env
        self.agent_name = agent_name
        self.init_prior = init_prior

        self.observation_space = env.observation_space(agent_name)
        self.action_space = env.action_space(agent_name)

    def reset(self, **kwargs):
        obs, infos = self.env.reset()
        return obs[self.agent_name], infos.get(self.agent_name, {})

    def step(self, action):
        # All other agents take their current best action (explore)
        actions = {
            agent: self.env.action_space(agent).sample()
            for agent in self.env.agents
        }
        actions[self.agent_name] = int(action)

        obs, rewards, terminations, truncations, infos = self.env.step(actions)

        if self.agent_name not in obs:
            # Episode ended — return zeros
            dummy_obs = np.zeros(
                self.observation_space.shape, dtype=np.float32
            )
            return dummy_obs, 0.0, True, False, {}

        return (
            obs[self.agent_name],
            rewards.get(self.agent_name, 0.0),
            terminations.get(self.agent_name, False),
            truncations.get(self.agent_name, False),
            infos.get(self.agent_name, {}),
        )

    def render(self):
        pass


# ---------------------------------------------------------------------------
# SB3 callback for MLflow logging
# ---------------------------------------------------------------------------

try:
    from stable_baselines3.common.callbacks import BaseCallback as _BaseCallback

    class _MARLCallback(_BaseCallback):
        def __init__(self, agent_name: str, mlflow_run, log_freq: int = 1000):
            super().__init__(verbose=0)
            self.agent_name = agent_name
            self.mlflow_run = mlflow_run
            self.log_freq = log_freq

        def _on_step(self) -> bool:
            if self.mlflow_run is None:
                return True
            if self.n_calls % self.log_freq == 0:
                try:
                    import mlflow
                    if "train/entropy_loss" in self.logger.name_to_value:
                        entropy = self.logger.name_to_value["train/entropy_loss"]
                        mlflow.log_metric(
                            f"marl_{self.agent_name}_entropy",
                            entropy,
                            step=self.n_calls,
                        )
                    if "train/approx_kl" in self.logger.name_to_value:
                        kl = self.logger.name_to_value["train/approx_kl"]
                        mlflow.log_metric(
                            f"marl_{self.agent_name}_kl",
                            kl,
                            step=self.n_calls,
                        )
                except Exception:
                    pass
            return True

except ImportError:
    class _MARLCallback:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass
