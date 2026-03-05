"""
Stage 2A: Behavioral Cloning + MaxEnt IRL

Behavioral Cloning: Supervised MLP that learns the expert policy from
state-action pairs extracted from clinical recordings.

MaxEnt IRL: Gradient ascent on the expected reward under the expert policy
minus the expected reward under the current policy, with importance sampling
for the partition function Z_θ.

Outputs: gradient magnitudes per biomarker → used as action priors for MARL.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.error("PyTorch not available.")


# ---------------------------------------------------------------------------
# Behavioral Cloning MLP
# ---------------------------------------------------------------------------

class BehavioralCloningMLP(nn.Module):
    """
    3-layer MLP: 152 → 256 → 128 → N_BIOMARKERS (9)
    Predicts the expert action (which biomarker subset to select) from state.
    """

    def __init__(self, state_dim: int = 152, n_actions: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


# ---------------------------------------------------------------------------
# Reward MLP (used by IRL)
# ---------------------------------------------------------------------------

class RewardMLP(nn.Module):
    """
    2-layer MLP: D_MODEL → 64 → 1 (scalar reward estimate)
    Input is the 128-dim encoder embedding.
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x).squeeze(-1)  # [B]


# ---------------------------------------------------------------------------
# Expert state sequences
# ---------------------------------------------------------------------------

def build_expert_sequences(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cfg,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (state, action) pairs from embeddings + binary labels.
    State: embedding padded/concatenated to STATE_DIM.
    Action: binary label (simplified as scalar action → biomarker index).

    Returns:
        states: [N, STATE_DIM]
        actions: [N, N_BIOMARKERS] — one-hot encoded expert action
    """
    N, D = embeddings.shape
    state_dim = cfg.STATE_DIM

    # Pad embeddings to STATE_DIM with zeros
    if D < state_dim:
        pad = np.zeros((N, state_dim - D), dtype=np.float32)
        states = np.concatenate([embeddings.astype(np.float32), pad], axis=1)
    else:
        states = embeddings[:, :state_dim].astype(np.float32)

    # Expert action: for pathological samples, activate all biomarkers.
    # For normal samples, activate none. This gives the IRL a supervision signal.
    n_bio = cfg.N_BIOMARKERS
    actions = np.zeros((N, n_bio), dtype=np.float32)
    patho_mask = labels > 0
    actions[patho_mask] = 1.0

    return states, actions


# ---------------------------------------------------------------------------
# Behavioral Cloning
# ---------------------------------------------------------------------------

def fit_behavioral_cloning(
    expert_states: np.ndarray,
    expert_actions: np.ndarray,
    cfg,
    mlflow_run=None,
    device: Optional[str] = None,
) -> "BehavioralCloningMLP":
    """
    Train BC-MLP with cross-entropy loss on expert (state, action) pairs.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch required")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BehavioralCloningMLP(
        state_dim=cfg.STATE_DIM, n_actions=cfg.N_BIOMARKERS
    ).to(device)

    x_t = torch.tensor(expert_states, dtype=torch.float32)
    y_t = torch.tensor(expert_actions, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(x_t, y_t),
        batch_size=cfg.BC_BATCH,
        shuffle=True,
        drop_last=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.BC_LR)
    model.train()

    for epoch in range(1, cfg.BC_EPOCHS + 1):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            # Binary cross-entropy per biomarker, mean over batch
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 20 == 0 or epoch == cfg.BC_EPOCHS:
            avg = epoch_loss / max(len(loader), 1)
            logger.info("BC epoch %d/%d — loss=%.4f", epoch, cfg.BC_EPOCHS, avg)
            if mlflow_run is not None:
                try:
                    import mlflow
                    mlflow.log_metric("bc_loss", avg, step=epoch)
                except Exception:
                    pass

    # Save artifact
    ckpt_path = cfg.CHECKPOINT_DIR / "bc_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    if mlflow_run is not None:
        try:
            import mlflow
            mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")
        except Exception:
            pass

    logger.info("BC model saved to %s", ckpt_path)
    return model


# ---------------------------------------------------------------------------
# MaxEnt IRL
# ---------------------------------------------------------------------------

def maxent_irl_train(
    expert_embeddings: np.ndarray,
    all_embeddings: np.ndarray,
    cfg,
    mlflow_run=None,
    device: Optional[str] = None,
) -> "RewardMLP":
    """
    Maximum Entropy IRL via gradient ascent:
        ∇L = E_expert[∇R_θ] - E_policy[∇R_θ]

    Partition function Z_θ estimated via importance sampling over all_embeddings.

    Returns trained RewardMLP.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch required")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    reward_model = RewardMLP(d_model=cfg.D_MODEL).to(device)
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=cfg.IRL_LR)

    expert_t = torch.tensor(expert_embeddings, dtype=torch.float32).to(device)
    all_t = torch.tensor(all_embeddings, dtype=torch.float32).to(device)

    prev_loss = float("inf")
    reward_model.train()

    for iteration in range(1, cfg.IRL_MAX_ITER + 1):
        # E_expert[R]
        r_expert = reward_model(expert_t).mean()

        # E_policy[R] via importance sampling
        # Importance weights: softmax(R) — normalizes like partition function
        with torch.no_grad():
            r_all = reward_model(all_t)
            log_Z = torch.logsumexp(r_all, dim=0)

        r_policy = reward_model(all_t)
        weights = torch.exp(r_policy - log_Z).detach()
        r_policy_weighted = (weights * r_policy).sum()

        # MaxEnt IRL loss: maximize (E_expert - E_policy)
        loss = -(r_expert - r_policy_weighted)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        delta = abs(prev_loss - loss_val)

        if iteration % 50 == 0 or iteration == cfg.IRL_MAX_ITER:
            logger.info(
                "IRL iter %d/%d — loss=%.6f  Δ=%.2e",
                iteration, cfg.IRL_MAX_ITER, loss_val, delta
            )
            if mlflow_run is not None:
                try:
                    import mlflow
                    mlflow.log_metric("irl_loss", loss_val, step=iteration)
                except Exception:
                    pass

        if delta < cfg.IRL_EARLY_STOP and iteration > 10:
            logger.info("IRL early stop at iteration %d (Δloss=%.2e)", iteration, delta)
            break

        prev_loss = loss_val

    ckpt_path = cfg.CHECKPOINT_DIR / "reward_model.pt"
    torch.save(reward_model.state_dict(), ckpt_path)
    logger.info("Reward model saved to %s", ckpt_path)

    return reward_model


# ---------------------------------------------------------------------------
# Gradient magnitude extraction
# ---------------------------------------------------------------------------

def compute_gradient_magnitudes(
    reward_model: "RewardMLP",
    all_embeddings: np.ndarray,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Compute mean gradient magnitude of reward w.r.t. each input dimension,
    then aggregate across D_MODEL dimensions into N_BIOMARKERS approximate scores
    via equal partitioning.

    Returns: [N_BIOMARKERS] gradient magnitude array.

    Note: embeddings are D_MODEL=128 dimensional; we evenly divide the gradient
    vector across N_BIOMARKERS=9 chunks and take the mean norm per chunk.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch required")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    reward_model = reward_model.to(device)
    reward_model.eval()

    x_t = torch.tensor(all_embeddings, dtype=torch.float32, requires_grad=False).to(device)

    # Compute Jacobian via batched grad
    grad_mags = []
    batch_size = 64
    for i in range(0, len(x_t), batch_size):
        batch = x_t[i: i + batch_size].clone().requires_grad_(True)
        r = reward_model(batch)
        r.sum().backward()
        grad_mags.append(batch.grad.abs().cpu().numpy())

    all_grads = np.concatenate(grad_mags, axis=0)  # [N, D_MODEL]
    mean_grad = all_grads.mean(axis=0)              # [D_MODEL]

    # Partition D_MODEL dimensions across N_BIOMARKERS
    d = len(mean_grad)
    n_bio = 9
    chunk = max(1, d // n_bio)
    bio_magnitudes = np.zeros(n_bio, dtype=np.float32)
    for i in range(n_bio):
        start = i * chunk
        end = min((i + 1) * chunk, d)
        bio_magnitudes[i] = mean_grad[start:end].mean()

    # Normalize to [0, 1]
    bio_range = bio_magnitudes.max() - bio_magnitudes.min()
    if bio_range > 0:
        bio_magnitudes = (bio_magnitudes - bio_magnitudes.min()) / bio_range

    return bio_magnitudes


def get_marl_priors(
    grad_magnitudes: np.ndarray,
    biomarker_names: List[str],
) -> Dict[int, np.ndarray]:
    """
    Compute per-agent action prior probabilities from IRL gradient magnitudes.

    Returns dict: {agent_index: [p_exclude, p_include]} for each of 9 agents.

    Special rule:
    - angular_jerk (index 8): forced to [0.1, 0.9] (always prefer include)
    - Logs warning if angular_jerk gradient is in top-3.
    """
    n = len(biomarker_names)
    priors: Dict[int, np.ndarray] = {}

    # Top-3 by gradient magnitude
    top3 = np.argsort(grad_magnitudes)[-3:]
    angular_jerk_idx = biomarker_names.index("angular_jerk") if "angular_jerk" in biomarker_names else 8

    if angular_jerk_idx in top3:
        logger.warning(
            "angular_jerk (index %d) is in top-3 gradient magnitudes. "
            "IRL signal aligns with clinical prior — good.", angular_jerk_idx
        )

    for i in range(n):
        if i == angular_jerk_idx:
            # Force include for angular_jerk regardless of gradient
            priors[i] = np.array([0.1, 0.9], dtype=np.float32)
        else:
            # p_include proportional to gradient magnitude
            p_include = float(np.clip(grad_magnitudes[i], 0.1, 0.9))
            priors[i] = np.array([1.0 - p_include, p_include], dtype=np.float32)

    return priors
