"""
Stage 1: Contrastive Transformer Encoder with NT-Xent loss.

Architecture:
- Linear projection 9 → D_MODEL
- Learned positional encoding
- 4-layer transformer (8 heads, d_model=128)
- Mean-pool over time → 128-dim embedding

Pre-training objective: NT-Xent contrastive loss on augmented window pairs.
GATE: linear probe AUROC must be >0.70 after pre-training.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional, Tuple

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
    logger.error("PyTorch not available. Install torch first.")


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment_window(x: "torch.Tensor") -> "torch.Tensor":
    """
    Apply stochastic augmentation to a window tensor [W, C].
    - Temporal jitter ±3 frames (roll)
    - Gaussian noise N(0, 0.01)
    - Random zero-out of 2 of 9 channels
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch required")

    # Temporal jitter
    shift = torch.randint(-3, 4, (1,)).item()
    x = torch.roll(x, shifts=int(shift), dims=0)

    # Gaussian noise
    x = x + 0.01 * torch.randn_like(x)

    # Channel dropout (zero out 2 random channels)
    n_channels = x.shape[-1]
    drop_idx = torch.randperm(n_channels)[:2]
    x = x.clone()
    x[:, drop_idx] = 0.0

    return x


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: [B, T, D]
        T = x.shape[1]
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # [1, T]
        return x + self.embedding(positions)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class ContrastiveTransformerEncoder(nn.Module):
    """
    4-layer transformer encoder.
    Input:  [B, W, N_BIOMARKERS]
    Output: [B, D_MODEL]  (128-dim embedding)
    """

    def __init__(
        self,
        n_biomarkers: int = 9,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        window_size: int = 30,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_biomarkers, d_model)
        self.pos_enc = LearnedPositionalEncoding(d_model, max_len=window_size + 10)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.d_model = d_model

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: [B, W, C]
        x = self.input_proj(x)          # [B, W, D]
        x = self.pos_enc(x)             # [B, W, D]
        x = self.transformer(x)         # [B, W, D]
        x = x.mean(dim=1)              # [B, D]  — mean-pool over time
        assert x.shape[-1] == self.d_model, (
            f"Encoder output dim mismatch: got {x.shape[-1]}, expected {self.d_model}"
        )
        return x


# ---------------------------------------------------------------------------
# NT-Xent Loss
# ---------------------------------------------------------------------------

class NT_XentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross-Entropy loss.
    Operates on paired embeddings (z_i, z_j) from the same window.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, z_i: "torch.Tensor", z_j: "torch.Tensor"
    ) -> "torch.Tensor":
        # z_i, z_j: [B, D]
        B = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)   # [2B, D]
        z = F.normalize(z, dim=1)

        sim = torch.mm(z, z.T) / self.temperature  # [2B, 2B]

        # Mask out self-similarity
        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, float("-inf"))

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat(
            [torch.arange(B, 2 * B), torch.arange(0, B)], dim=0
        ).to(z.device)

        loss = F.cross_entropy(sim, labels)
        return loss


# ---------------------------------------------------------------------------
# Linear probe for AUROC gate
# ---------------------------------------------------------------------------

def linear_probe_auroc(
    encoder: "ContrastiveTransformerEncoder",
    windows: np.ndarray,
    labels: np.ndarray,
    device: Optional[str] = None,
) -> float:
    """
    Train a simple logistic regression head on frozen encoder embeddings.
    Returns binary AUROC (pathological vs. normal).
    labels are expected as multi-class; pathological = label > 0.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch required")

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except ImportError as e:
        raise ImportError("scikit-learn required for linear probe") from e

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = encoder.to(device)
    encoder.eval()

    x_tensor = torch.tensor(windows, dtype=torch.float32)
    dataset = TensorDataset(x_tensor)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    embeddings = []
    with torch.no_grad():
        for (batch,) in loader:
            emb = encoder(batch.to(device))
            embeddings.append(emb.cpu().numpy())

    X = np.concatenate(embeddings, axis=0)
    y_binary = (labels > 0).astype(int)

    # Need at least 2 classes
    if len(np.unique(y_binary)) < 2:
        logger.warning("Only one class present in probe labels — returning AUROC=0.5")
        return 0.5

    # Train/val split for probe (80/20)
    split = int(0.8 * len(X))
    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(X[:split], y_binary[:split])
    probs = clf.predict_proba(X[split:])[:, 1]
    auroc = roc_auc_score(y_binary[split:], probs)
    return float(auroc)


# ---------------------------------------------------------------------------
# Pre-training loop
# ---------------------------------------------------------------------------

def pretrain_encoder(
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    cfg,
    mlflow_run=None,
    device: Optional[str] = None,
) -> "ContrastiveTransformerEncoder":
    """
    Pre-train encoder with NT-Xent on augmented window pairs.
    Saves checkpoint to cfg.CHECKPOINT_DIR / 'encoder_pretrained.pt'.
    GATE: linear probe AUROC > cfg.ENCODER_MIN_AUROC (default 0.70).
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch required for encoder pre-training")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Pre-training encoder on %s (device=%s)", train_windows.shape, device)

    encoder = ContrastiveTransformerEncoder(
        n_biomarkers=cfg.N_BIOMARKERS,
        d_model=cfg.D_MODEL,
        n_heads=cfg.N_HEADS,
        n_layers=cfg.N_LAYERS,
        window_size=cfg.WINDOW_SIZE,
    ).to(device)

    loss_fn = NT_XentLoss(temperature=cfg.ENCODER_TEMPERATURE)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg.ENCODER_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.ENCODER_EPOCHS
    )

    x_tensor = torch.tensor(train_windows, dtype=torch.float32)
    dataset = TensorDataset(x_tensor)
    loader = DataLoader(
        dataset, batch_size=cfg.ENCODER_BATCH, shuffle=True, drop_last=True
    )

    encoder.train()
    for epoch in range(1, cfg.ENCODER_EPOCHS + 1):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            # Create two augmented views
            z_i = encoder(torch.stack([augment_window(b) for b in batch]))
            z_j = encoder(torch.stack([augment_window(b) for b in batch]))
            loss = loss_fn(z_i, z_j)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / max(len(loader), 1)

        if epoch % 10 == 0 or epoch == cfg.ENCODER_EPOCHS:
            logger.info("Encoder epoch %d/%d — loss=%.4f", epoch, cfg.ENCODER_EPOCHS, avg_loss)
            if mlflow_run is not None:
                try:
                    import mlflow
                    mlflow.log_metric("encoder_loss", avg_loss, step=epoch)
                except Exception:
                    pass

    # Save checkpoint
    ckpt_path = cfg.CHECKPOINT_DIR / "encoder_pretrained.pt"
    torch.save(encoder.state_dict(), ckpt_path)
    logger.info("Encoder checkpoint saved to %s", ckpt_path)

    # GATE: linear probe AUROC
    auroc = linear_probe_auroc(encoder, train_windows, train_labels, device=device)
    logger.info("Linear probe AUROC = %.4f (threshold=%.2f)", auroc, cfg.ENCODER_MIN_AUROC)

    if mlflow_run is not None:
        try:
            import mlflow
            mlflow.log_metric("encoder_linear_probe_auroc", auroc)
            mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")
        except Exception:
            pass

    if auroc < cfg.ENCODER_MIN_AUROC:
        raise ValueError(
            f"Encoder linear probe AUROC={auroc:.4f} < gate threshold "
            f"{cfg.ENCODER_MIN_AUROC}. Increase ENCODER_EPOCHS or check data quality."
        )

    return encoder


def load_encoder(
    cfg, device: Optional[str] = None
) -> "ContrastiveTransformerEncoder":
    """Load pre-trained encoder from checkpoint."""
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch required")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = cfg.CHECKPOINT_DIR / "encoder_pretrained.pt"
    encoder = ContrastiveTransformerEncoder(
        n_biomarkers=cfg.N_BIOMARKERS,
        d_model=cfg.D_MODEL,
        n_heads=cfg.N_HEADS,
        n_layers=cfg.N_LAYERS,
        window_size=cfg.WINDOW_SIZE,
    )
    encoder.load_state_dict(torch.load(ckpt_path, map_location=device))
    encoder.to(device)
    encoder.eval()
    return encoder


def encode_windows(
    encoder: "ContrastiveTransformerEncoder",
    windows: np.ndarray,
    device: Optional[str] = None,
    batch_size: int = 128,
) -> np.ndarray:
    """
    Encode windows → embeddings [N, D_MODEL].
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch required")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = encoder.to(device)
    encoder.eval()

    x_tensor = torch.tensor(windows, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=False)

    parts = []
    with torch.no_grad():
        for (batch,) in loader:
            emb = encoder(batch.to(device))
            parts.append(emb.cpu().numpy())

    return np.concatenate(parts, axis=0)
