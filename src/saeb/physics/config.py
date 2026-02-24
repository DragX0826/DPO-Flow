import torch
import numpy as np

class ForceFieldParameters:
    """
    Stores parameters for the differentiable force field.
    Includes atom-specific VdW radii, bond constants, etc.
    """
    def __init__(self, device='cpu', no_physics=False, no_hsa=False):
        # Atomic Radii (Angstroms) for C, N, O, S, F, P, Cl, Br, I
        self.vdw_radii = torch.tensor([1.7, 1.55, 1.52, 1.8, 1.47, 1.8, 1.75, 1.85, 1.98], device=device)
        self.epsilon = torch.tensor([0.1, 0.1, 0.15, 0.2, 0.1, 0.2, 0.2, 0.2, 0.3], device=device)
        self.standard_valencies = torch.tensor([4, 3, 2, 2, 1, 3, 1, 1, 1], device=device).float()
        
        # [v97.9 Normalization] Standard Force Field Constants
        self.bond_length_mean = 1.45 # Standard C-C distance proxy
        self.bond_k = 100.0 # Reduced from 500 to prevent stiffness-shocks
        self.angle_mean = np.deg2rad(109.5)
        self.angle_k = 20.0 # Reduced from 100
        self.softness_start = 5.0
        self.softness_end = 0.0
        self.no_physics = no_physics
        self.no_hsa = no_hsa
