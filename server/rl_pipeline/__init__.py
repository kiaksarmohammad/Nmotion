"""
Neuromotion-AI — Three-Stage Offline RL Pipeline
=================================================

Stage 1: Contrastive Transformer Encoder (NT-Xent pre-training)
Stage 2A: Behavioral Cloning + MaxEnt IRL
Stage 2B: MARL feature selection (PettingZoo + SB3 PPO)
Stage 3: SAC threshold/weight optimization (SB3 + HER)

Entry point: python server/rl_pipeline/train.py
"""

from .config import PipelineConfig
from .rubric import NeonatalRubric

__all__ = ["PipelineConfig", "NeonatalRubric"]
