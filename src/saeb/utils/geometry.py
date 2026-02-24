import torch
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def calculate_kabsch_rmsd(P, Q):
    """
    Standard RMSD without atom reordering (Kabsch Algorithm).
    Checks topology preservation.
    P, Q: (N, 3)
    """
    if P.shape[0] != Q.shape[0]:
        return 99.99
    
    try:
        # Center
        P_c = P - P.mean(dim=0)
        Q_c = Q - Q.mean(dim=0)
        
        # Covariance matrix
        H = torch.mm(P_c.t(), Q_c)
        
        # SVD (Bug Fix 12: Use linalg.svd which is stable/modern)
        U, S, Vh = torch.linalg.svd(H)
        
        # Vh from linalg.svd is V.mH (transpose for real matrices)
        # So V = Vh.t()
        V = Vh.t()

        # Rotation matrix calculation
        # R = V @ U.t()
        d = torch.det(V @ U.t())
        
        # Check for reflection
        if d < 0:
            V[:, -1] *= -1
            
        R = torch.mm(V, U.t())
            
        # Rotate P
        P_rot = torch.mm(P_c, R.t())
        
        # RMSD
        diff = P_rot - Q_c
        rmsd = torch.sqrt((diff ** 2).sum() / P.size(0))
        return rmsd.item()
    except Exception:
        return 99.9

def calculate_rmsd_hungarian(P, Q):
    """
    Permutation-invariant RMSD using Hungarian matching.
    Robust to atom ordering mismatch.
    """
    try:
        # Detach for scipy
        P_np = P.detach().cpu().numpy()
        Q_np = Q.detach().cpu().numpy()
        
        # Handle shape mismatch via truncation
        n = min(P_np.shape[0], Q_np.shape[0])
        P_np = P_np[:n]
        Q_np = Q_np[:n]
        
        # Distance Matrix
        dists = np.linalg.norm(P_np[:, None, :] - Q_np[None, :, :], axis=-1)
        
        # Optimal Assignment
        row_ind, col_ind = linear_sum_assignment(dists**2)
        
        # Refine Fix: Use explicit indices for clarity
        # row_ind corresponds to P, col_ind to Q
        P_matched = P[torch.from_numpy(row_ind).to(P.device)]
        Q_matched = Q[torch.from_numpy(col_ind).to(Q.device)]
        
        # Compute RMSD on ordered pairs
        return calculate_kabsch_rmsd(P_matched, Q_matched)
    except Exception as e:
        import logging
        logging.getLogger("SAEB-Flow").error(f"Hungarian RMSD failed: {e}")
        return 99.9

def calculate_internal_rmsd(pos_batch):
    """
    Calculates the mean pairwise RMSD within a batch of conformations.
    
    NOTE: This calculation uses translation-alignment but NOT rotation-alignment.
    It is suitable for quick ensemble diversity checks but not for rigorous RMSD metrics.
    
    pos_batch: (B, N, 3)
    Returns: float (Average RMSD)
    """
    B, N, _ = pos_batch.shape
    if B < 2: return 0.0
    
    # Translation alignment: center each member of the batch
    pos_batch = pos_batch - pos_batch.mean(dim=1, keepdim=True)
    
    # Pairwise differences: (B, 1, N, 3) - (1, B, N, 3) -> (B, B, N, 3)
    diff = pos_batch.unsqueeze(1) - pos_batch.unsqueeze(0)
    dist_sq = diff.pow(2).sum(dim=-1) # (B, B, N)
    rmsd_mat = torch.sqrt(dist_sq.mean(dim=-1) + 1e-8) # (B, B)
    
    # Exclude diagonal (self-comparison)
    mask = ~torch.eye(B, dtype=torch.bool, device=pos_batch.device)
    valid_elements = rmsd_mat[mask]
    return valid_elements.mean().item() if valid_elements.numel() > 0 else 0.0

def rodrigues_rotation(axis, angle):
    """
    Vectorized Rodrigues rotation matrix formula.
    axis: (B, 3), angle: (B, 1) -> R: (B, 3, 3)
    """
    B = axis.shape[0]
    axis = torch.nn.functional.normalize(axis, dim=-1)
    cos_a = torch.cos(angle).view(B, 1, 1)
    sin_a = torch.sin(angle).view(B, 1, 1)
    
    # K matrix (cross-product matrix)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    K = torch.zeros(B, 3, 3, device=axis.device, dtype=axis.dtype)
    K[:, 0, 1] = -z; K[:, 0, 2] = y
    K[:, 1, 0] = z;  K[:, 1, 2] = -x
    K[:, 2, 0] = -y; K[:, 2, 1] = x
    
    # R = I + sin(a)K + (1-cos(a))K^2
    I = torch.eye(3, device=axis.device, dtype=axis.dtype).unsqueeze(0).repeat(B, 1, 1)
    K2 = torch.bmm(K, K)
    R = I + sin_a * K + (1 - cos_a) * K2
    return R
