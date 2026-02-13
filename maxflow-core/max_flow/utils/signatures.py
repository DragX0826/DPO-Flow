import torch
import torch.nn as nn

def signature_transform(path, depth=2):
    """
    Computes the Signature of a path in pure PyTorch.
    Standard signature: sequence of iterated integrals.
    path: (N, D) where N is path length, D is dimension.
    """
    N, D = path.shape
    increments = torch.diff(path, dim=0) # (N-1, D)
    
    # Order 1: increments sum
    sig = [torch.ones(1, device=path.device)] # Zeroth order
    sig.append(increments.sum(dim=0)) # First order (Total displacement)
    
    if depth >= 2:
        # Order 2: Iterated integrals (Double sum)
        # S_{i,j} = sum_{k < m} dx_i[k] * dx_j[m]
        # We can use outer product and prefix sums
        outer = torch.einsum('id,ie->ide', increments, increments)
        # This is a bit complex for pure torch without specialized kernels
        # Simple version using double sum:
        acc = torch.zeros(D, D, device=path.device)
        running_sum = torch.zeros(D, device=path.device)
        for i in range(N-1):
            acc += torch.outer(running_sum, increments[i])
            running_sum += increments[i]
        sig.append(acc.flatten())
        
    return torch.cat(sig)

class LeadLagTransform(nn.Module):
    """
    Transforms a path X into a Lead-Lag path (X_lead, X_lag).
    This allows the signature to capture the quadratic variation (volatility).
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: (N, D)
        n, d = x.shape
        # Interleave lead and lag
        # lead: [x0, x1, x1, x2, x2, ..., xn]
        # lag:  [x0, x0, x1, x1, x2, ..., xn-1]
        lead = torch.repeat_interleave(x, 2, dim=0)[1:]
        lag = torch.repeat_interleave(x, 2, dim=0)[:-1]
        
        # Result is (2N-1, D)
        # But Lead-Lag signature usually stacks them: (2N-1, 2D)
        ll_path = torch.cat([lead, lag], dim=-1)
        return ll_path

class SignatureLayer(nn.Module):
    """
    A learnable layer that extracts Log-Signatures from geometric paths.
    """
    def __init__(self, in_channels, depth=2, out_channels=64):
        super().__init__()
        self.depth = depth
        # Dimension of signature grows exponentially: sum(in_channels^k for k in 0..depth)
        sig_dim = 1 + in_channels
        if depth >= 2:
            sig_dim += in_channels ** 2
            
        self.proj = nn.Linear(sig_dim, out_channels)
        self.ll = LeadLagTransform()

    def forward(self, x):
        # x: (N, D)
        ll_path = self.ll(x)

# ============== Phase 22+: Log-Signature (Compressed) ==============

def log_signature_transform(path, depth=2):
    """
    Compute the Log-Signature of a path.
    
    Log-Signature is a compressed representation that lives in the free Lie algebra.
    It has exponentially smaller dimension than the full signature while preserving
    key path information.
    
    For depth=2, log-signature ≈ [S^1, S^2 - 0.5*(S^1 ⊗ S^1)]
    where S^k is the k-th level signature term.
    
    This uses a simplified BCH (Baker-Campbell-Hausdorff) approximation.
    """
    N, D = path.shape
    increments = torch.diff(path, dim=0)  # (N-1, D)
    
    # Level 1: Total displacement (same as signature)
    S1 = increments.sum(dim=0)  # (D,)
    
    log_sig = [torch.ones(1, device=path.device), S1]
    
    if depth >= 2:
        # Level 2 signature
        acc = torch.zeros(D, D, device=path.device)
        running_sum = torch.zeros(D, device=path.device)
        for i in range(N-1):
            acc += torch.outer(running_sum, increments[i])
            running_sum += increments[i]
        S2 = acc  # (D, D)
        
        # Log-Signature level 2: S2 - 0.5 * (S1 ⊗ S1)
        # The antisymmetric part
        log_S2 = S2 - 0.5 * torch.outer(S1, S1)
        
        # For log-signature, we only need the antisymmetric (Lie bracket) part
        # Dimension: D*(D-1)/2 instead of D*D
        # Extract upper triangular minus lower triangular
        antisym = log_S2 - log_S2.T
        # Flatten upper triangular
        indices = torch.triu_indices(D, D, offset=1)
        log_S2_compressed = antisym[indices[0], indices[1]]
        
        log_sig.append(log_S2_compressed)
    
    return torch.cat(log_sig)


class LogSignatureLayer(nn.Module):
    """
    Log-Signature Layer - Compressed Path Representation
    
    Advantages over full Signature:
    - Exponentially smaller dimension for high depths
    - Preserves essential path geometry
    - More efficient for deep networks
    """
    def __init__(self, in_channels, depth=2, out_channels=64):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        
        # Lead-Lag doubles the channels
        doubled = in_channels * 2
        
        # Log-signature dimension is much smaller
        # Level 0: 1
        # Level 1: doubled
        # Level 2: doubled*(doubled-1)/2 (antisymmetric part only)
        # 3D path: doubled=6. sig_dim=1+6+36=43. log_sig_dim=1+6+15=22. ~50% reduction.
        log_sig_dim = 1 + doubled
        if depth >= 2:
            log_sig_dim += (doubled * (doubled - 1)) // 2
            
        self.proj = nn.Linear(log_sig_dim, out_channels)
        self.ll = LeadLagTransform()

    def forward(self, x):
        """
        x: (N, D) - Path of N points in D dimensions
        """
        ll_path = self.ll(x)
        try:
            log_sig = log_signature_transform(ll_path, depth=self.depth)
        except Exception:
            # Fallback if dimensions are too small or path length is 1
            # Just use zero padding or flattened path
            log_sig = torch.zeros(1, self.proj.in_features, device=x.device)
            
        return self.proj(log_sig)
    
    def compression_ratio(self, D):
        """Compute compression ratio vs full signature."""
        doubled = D * 2
        full_sig_dim = 1 + doubled + doubled**2
        log_sig_dim = 1 + doubled + (doubled * (doubled - 1)) // 2
        return full_sig_dim / log_sig_dim

# ============== End Log-Signature ==============

