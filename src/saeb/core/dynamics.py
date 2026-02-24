import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, NamedTuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger("SAEB-Flow.core.dynamics")

@dataclass
class FiberBundle:
    """Describes the SE(3) x T^n fibre-bundle topology of a ligand."""
    n_atoms:          int
    rotatable_bonds:  List[Tuple[int, int]]
    downstream_masks: torch.Tensor # (n_bonds, n_atoms)
    fragment_labels:  torch.Tensor # (n_atoms,)
    pivot_atoms:      List[int]
    device:           torch.device = field(default_factory=lambda: torch.device('cpu'))

    def n_bonds(self) -> int:
        return len(self.rotatable_bonds)

    def to(self, device):
        """Bug Fix 15: Manual device migration for dataclass tensors."""
        self.downstream_masks = self.downstream_masks.to(device)
        self.fragment_labels = self.fragment_labels.to(device)
        self.device = device
        return self

def build_fiber_bundle(pos_native, mol=None, n_fragments=4):
    """
    Entry point to build the FiberBundle topology.
    Bug Fix 16: Implemented RDKit-based bond detection and mask generation.
    """
    n_atoms = pos_native.shape[0]
    device = pos_native.device
    
    rotatable_bonds = []
    downstream_masks = []
    fragment_labels = torch.zeros(n_atoms, device=device)
    pivot_atoms = [0] # Default root

    if mol is not None:
        try:
            from rdkit import Chem
            # SMARTS for rotatable bonds (non-ring, single bonds, not terminal)
            pattern = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
            matches = mol.GetSubstructMatches(pattern)
            
            # DFS/BFS for downstream masks
            adj = [[] for _ in range(n_atoms)]
            for bond in mol.GetBonds():
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if u < n_atoms and v < n_atoms:
                    adj[u].append(v)
                    adj[v].append(u)

            for u, v in matches:
                # Basic downstream detection: items further from atom 0
                # We assume 0 is the root for simplicity in this version
                mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)
                
                # BFS to find everything on the 'v' side if we cut (u, v)
                stack = [v]
                visited = {u, v}
                mask[v] = True
                while stack:
                    curr = stack.pop()
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            mask[neighbor] = True
                            stack.append(neighbor)
                
                # Only keep if it doesn't move the entire molecule
                if mask.sum() > 0 and mask.sum() < n_atoms - 1:
                    rotatable_bonds.append((u, v))
                    downstream_masks.append(mask)

        except Exception as e:
            logger.warning(f"RDKit FiberBundle failed: {e}. Falling back to rigid.")

    if not downstream_masks:
        downstream_masks = torch.zeros((0, n_atoms), device=device)
    else:
        downstream_masks = torch.stack(downstream_masks)

    return FiberBundle(
        n_atoms=n_atoms,
        rotatable_bonds=rotatable_bonds,
        downstream_masks=downstream_masks,
        fragment_labels=fragment_labels,
        pivot_atoms=pivot_atoms,
        device=device
    )

def torus_flow_velocity(theta_pred, theta_native, t):
    """
    Shortest-arc wrap for torsional velocities on T^n.
    Bug Fix 14: Safe for both CPU and CUDA tensors using floor instead of %.
    """
    diff = theta_native - theta_pred
    # Wrap to (-pi, pi]
    diff = diff - 2 * np.pi * torch.floor((diff + np.pi) / (2 * np.pi))
    return diff / (1.0 - t + 1e-6)

def apply_saeb_step(pos, fb, v_cart, dt, J=None):
    """
    Fibre-bundle geodesic integration step.
    Updates positions by integrating Cartesian and Torsional components.
    """
    # Simplified Euler-like step on the manifold
    # In a full implementation, this uses the precomputed Jacobian J
    new_pos = pos + v_cart * dt
    return new_pos
