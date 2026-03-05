"""
PipelineConfig — single source of truth for all hyperparameters, paths, and constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class PipelineConfig:
    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    DATA_DIR: Path = field(default_factory=lambda: Path("data"))
    CHECKPOINT_DIR: Path = field(default_factory=lambda: Path("output/checkpoints"))
    OUTPUT_DIR: Path = field(default_factory=lambda: Path("output"))
    MLFLOW_URI: str = "mlruns"

    # ------------------------------------------------------------------
    # MDP / Feature space
    # ------------------------------------------------------------------
    STATE_DIM: int = 152           # flattened state for SAC (see sac_optimizer.py)
    N_BIOMARKERS: int = 9
    BIOMARKER_NAMES: List[str] = field(default_factory=lambda: [
        "entropy",          # 0
        "fractal_dim",      # 1
        "kinetic_energy",   # 2
        "root_stress",      # 3
        "phase_x",          # 4
        "phase_v",          # 5
        "mean_velocity",    # 6
        "peak_frequency",   # 7
        "angular_jerk",     # 8  ← forced prior by IRL
    ])

    # ------------------------------------------------------------------
    # Stage 1 — Contrastive Encoder
    # ------------------------------------------------------------------
    D_MODEL: int = 128
    N_HEADS: int = 8
    N_LAYERS: int = 4
    WINDOW_SIZE: int = 30
    STRIDE: int = 15
    ENCODER_EPOCHS: int = 200
    ENCODER_LR: float = 3e-4
    ENCODER_BATCH: int = 64
    ENCODER_TEMPERATURE: float = 0.07
    ENCODER_MIN_AUROC: float = 0.70   # linear-probe gate

    # ------------------------------------------------------------------
    # Stage 2A — IRL
    # ------------------------------------------------------------------
    IRL_LR: float = 1e-4
    IRL_MAX_ITER: int = 500
    IRL_EARLY_STOP: float = 1e-5
    BC_EPOCHS: int = 100
    BC_LR: float = 1e-3
    BC_BATCH: int = 32

    # ------------------------------------------------------------------
    # Stage 2B — MARL (SB3 independent PPO)
    # ------------------------------------------------------------------
    MARL_TOTAL_TIMESTEPS: int = 200_000
    MARL_PPO_LR: float = 3e-4
    MARL_MAX_STEPS: int = 500
    MARL_ENTROPY_THRESHOLD: float = 0.01

    # ------------------------------------------------------------------
    # Composite Reward weights
    # ------------------------------------------------------------------
    LAMBDA1: float = 0.4    # R_intermediate weight
    LAMBDA2: float = 0.1    # R_complexity weight
    LAMBDA3: float = 0.2    # R_plausibility weight
    FN_WEIGHT: float = 3.0  # false-negative penalty multiplier
    FP_WEIGHT: float = 1.0  # false-positive penalty multiplier

    # ------------------------------------------------------------------
    # Stage 3 — SAC
    # ------------------------------------------------------------------
    REPLAY_BUFFER: int = 50_000
    BATCH_SIZE: int = 256
    MAX_EPISODES: int = 2000
    SAC_LR: float = 3e-4
    SAC_TOTAL_TIMESTEPS: int = 500_000

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    OPE_VARIANTS: Dict[str, Dict] = field(default_factory=lambda: {
        "R1": {"lambda1": 0.4, "lambda2": 0.1, "lambda3": 0.2},
        "R2": {"lambda1": 0.3, "lambda2": 0.2, "lambda3": 0.3},
        "R3": {"lambda1": 0.5, "lambda2": 0.05, "lambda3": 0.15},
        "R4": {"lambda1": 0.2, "lambda2": 0.3, "lambda3": 0.1},
        "R5": {"lambda1": 0.4, "lambda2": 0.0, "lambda3": 0.4},
    })
    NULL_SHUFFLE_N: int = 100
    OPE_MAX_DEGRADATION: float = 0.20
    NULL_MAX_AUROC: float = 0.55
    SENSITIVITY_TARGET: float = 0.85

    # ------------------------------------------------------------------
    # Data splits
    # ------------------------------------------------------------------
    TRAIN_FRAC: float = 0.70
    VAL_FRAC: float = 0.15
    TEST_FRAC: float = 0.15
    SMOTE_TARGET_RATIO: float = 3.0

    # ------------------------------------------------------------------
    # Clinical priors — expected direction per biomarker for pathology
    # ------------------------------------------------------------------
    CLINICAL_DIRECTIONS: Dict[str, str] = field(default_factory=lambda: {
        "entropy": "down",
        "fractal_dim": "down",
        "kinetic_energy": "up",
        "root_stress": "up",
        "phase_x": "low_variance",
        "phase_v": "low_variance",
        "mean_velocity": "up",
        "peak_frequency": "up",
        "angular_jerk": "up",
    })

    def __post_init__(self):
        self.DATA_DIR = Path(self.DATA_DIR)
        self.CHECKPOINT_DIR = Path(self.CHECKPOINT_DIR)
        self.OUTPUT_DIR = Path(self.OUTPUT_DIR)
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_args(cls, args) -> "PipelineConfig":
        """Build config from argparse namespace, overriding defaults."""
        cfg = cls()
        if hasattr(args, "data_dir") and args.data_dir:
            cfg.DATA_DIR = Path(args.data_dir)
        if hasattr(args, "output_dir") and args.output_dir:
            cfg.OUTPUT_DIR = Path(args.output_dir)
            cfg.CHECKPOINT_DIR = cfg.OUTPUT_DIR / "checkpoints"
        if hasattr(args, "epochs") and args.epochs:
            cfg.ENCODER_EPOCHS = args.epochs
        cfg.__post_init__()
        return cfg
