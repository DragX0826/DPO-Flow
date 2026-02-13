from dataclasses import dataclass, field, asdict
import json
import os
from typing import Optional

@dataclass
class MaxRLConfig:
    """
    Central Configuration for MaxFlow Universal Engine.
    Supports JSON serialization for experiment tracking.
    """
    # Optimization
    lr: float = 3e-5
    batch_size: int = 16
    grad_clip_norm: float = 1.0 # SOTA Rescue Default
    beta: float = 0.1 # MaxRL KL penalty
    
    # SOTA Loss Components
    lambda_geom: float = 0.01
    lambda_anchor: float = 0.05  # Phase 63: Reduced 10x to unblock MaxRL gradient
    clip_val: float = 10.0 # SMaxRL
    
    # Phase 63: DrugCLIP-Inspired Contrastive Alignment
    lambda_clip: float = 0.1  # InfoNCE contrastive weight
    clip_temperature: float = 0.07  # Cosine similarity temperature (DrugCLIP default)
    lambda_atom: float = 0.5  # Categorical atom preference weight (SOTA Phase 65)
    
    # Dynamic Reward Scaling
    use_dynamic_scaling: bool = True
    reward_scale_initial: float = 20.0 # Fallback/Initial scale
    
    # System
    mixed_precision: str = "fp16" # "no", "fp16", "bf16"
    save_every: int = 1
    
    # HFT Efficiency & Precision (Phase 55)
    hft_mode: bool = False # Enable all HFT optimizations
    use_ema: bool = True
    ema_decay: float = 0.999
    compile_model: bool = False # torch.compile (Inductor)
    
    # Phase 11: SOTA Methodology Expansion
    use_maxrl: bool = False # Maximum Likelihood RL (Reward/Baseline reweighting)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=4)
            
    @classmethod
    def load(cls, path: str) -> 'MaxRLConfig':
        if not os.path.exists(path):
            print(f"Config file {path} not found. Using defaults.")
            return cls()
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
