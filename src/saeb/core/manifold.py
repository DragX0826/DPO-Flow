import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List
import logging

logger = logging.getLogger("SAEB-Flow.core.manifold")

def conditional_flow_target(pos_pred, pos_native, t):
    """
    Conditional Flow Matching target (Cartesian).
    Computes the velocity vector pointing towards the native pose.
    """
    # (B, N, 3) - (B, N, 3) or (N, 3)
    if pos_native.dim() == 2:
        pos_native = pos_native.unsqueeze(0).expand(pos_pred.size(0), -1, -1)
    
    # Conditional vector field: v(xt|x1) = (x1 - xt) / (1 - t)
    v_target = (pos_native - pos_pred) / (1.0 - t + 1e-6)
    return v_target

def saeb_harmonic_prior(fb, pos_native, p_center, B, noise_scale=5.0):
    """
    Harmonic prior sampling on the FiberBundle.
    Preserves internal rigid geometry while randomizing global orientation and torsions.
    """
    N = fb.n_atoms
    device = pos_native.device
    
    # Random global rotation
    # Random torsion angles
    # Center at pocket
    
    # Placeholder for full implementation
    pos_prior = p_center.view(1, 1, 3).expand(B, N, -1) + torch.randn(B, N, 3, device=device) * noise_scale
    return pos_prior

def sample_torsional_prior(fb, B):
    """Samples random angles for all rotatable bonds."""
    return torch.rand(B, fb.n_bonds()) * 2 * np.pi
