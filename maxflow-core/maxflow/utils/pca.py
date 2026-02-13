import torch

def compute_pca_rotation(pos_P, batch_P=None):
    """
    Computes the rotation matrix to align the protein pocket's principal axes to the global axes.
    
    Args:
        pos_P (Tensor): Protein atom positions (N_P, 3).
        batch_P (Tensor, optional): Batch indices for protein atoms. 
                                    If None, assumes single batch.
    
    Returns:
        rot_matrices (Tensor): Rotation matrices (B, 3, 3).
        centers (Tensor): Pocket centers (B, 3).
    """
    if batch_P is None:
        batch_P = torch.zeros(pos_P.size(0), dtype=torch.long, device=pos_P.device)
    
    unique_batches = torch.unique(batch_P)
    rot_matrices = []
    centers = []

    for b in unique_batches:
        mask = (batch_P == b)
        pos = pos_P[mask]
        
        # 1. Center
        center = pos.mean(dim=0)
        pos_centered = pos - center
        
        # 2. PCA via SVD
        # Covariance matrix approximation: X^T X
        # U, S, V = torch.svd(pos_centered)
        # However, we want the axes. V are the principal directions.
        # We want to rotate V to Identity. So R = V^T.
        
        try:
            # Add small noise for stability
            if pos_centered.size(0) < 3:
                U, S, V = torch.svd(torch.eye(3, device=pos.device))
            else:
                U, S, V = torch.svd(pos_centered)
                
            R = V.t() # (3, 3)
            
            # Ensure proper rotation (det=1) to avoid reflection
            if torch.det(R) < 0:
                V[:, -1] *= -1
                R = V.t()
                
            # Sign correction for axes (Skewness heuristic for determinism)
            # This fixes the sign flip ambiguity of PCA.
            # For each axis v_i, ensure sum((pos_centered @ v_i)^3) > 0
            for i in range(3):
                v_i = V[:, i] # Column i is the i-th principal axis
                # Projected points along this axis
                proj = torch.matmul(pos_centered, v_i)
                third_moment = torch.sum(proj ** 3)
                if third_moment < 0:
                    V[:, i] *= -1
            
            # Re-calculate R after sign correction
            R = V.t()
            
            # Final check for det=1 (might flip again if sign correction changed chirality)
            if torch.det(R) < 0:
                # If sign correction resulted in a reflection, flip the last axis back
                # to maintain a right-handed coordinate system.
                V[:, -1] *= -1
                R = V.t()
                
        except RuntimeError:
            R = torch.eye(3, device=pos.device)

        rot_matrices.append(R)
        centers.append(center)
        
    return torch.stack(rot_matrices), torch.stack(centers)

def apply_canonicalization(pos, batch, rot_matrices, centers):
    """
    Applies the canonical transformation: x' = R(x - center).
    
    Args:
        pos (Tensor): Atom positions (N, 3).
        batch (Tensor): Batch indices (N,).
        rot_matrices (Tensor): (B, 3, 3).
        centers (Tensor): (B, 3).
    """
    unique_batches = torch.unique(batch)
    pos_canonical = torch.zeros_like(pos)
    
    for i, b in enumerate(unique_batches):
        mask = (batch == b)
        p = pos[mask]
        c = centers[i]
        R = rot_matrices[i]
        
        # Center then Rotate
        p_centered = p - c
        p_rotated = torch.matmul(p_centered, R.t()) # (N, 3) x (3, 3)^T = (N, 3)
        pos_canonical[mask] = p_rotated
        
    return pos_canonical

def reverse_canonicalization(pos_canonical, batch, rot_matrices, centers):
    """
    Applies inverse transformation: x = R^T x' + center.
    """
    unique_batches = torch.unique(batch)
    pos_original = torch.zeros_like(pos_canonical)
    
    for i, b in enumerate(unique_batches):
        mask = (batch == b)
        p_prime = pos_canonical[mask]
        c = centers[i]
        R = rot_matrices[i]
        
        # Inverse Rotate then Translate
        # x = p_prime @ R + c
        p_unrotated = torch.matmul(p_prime, R) 
        p_original = p_unrotated + c
        pos_original[mask] = p_original
        
    return pos_original
