# max_flow/utils/scatter.py

import torch

def robust_scatter_mean(src, index, dim=0, dim_size=None):
    """
    Robust scatter_mean fallback when torch_scatter is not available.
    """
    try:
        from torch_scatter import scatter_mean
        return scatter_mean(src, index, dim=dim, dim_size=dim_size)
    except ImportError:
        if dim_size is None:
            dim_size = index.max().item() + 1
        
        # Flatten index for safety
        index = index.view(-1)
        
        # Initialize output
        out_shape = list(src.shape)
        out_shape[dim] = dim_size
        out = torch.zeros(*out_shape, device=src.device, dtype=src.dtype)
        
        # Sum
        out.index_add_(dim, index, src)
        
        # Count (scalar count per index)
        count = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
        ones_for_count = torch.ones(src.size(dim), device=src.device, dtype=src.dtype)
        count.index_add_(0, index, ones_for_count)
        
        # Reshape count for broadcasting
        reshaper = [1] * len(src.shape)
        reshaper[dim] = dim_size
        count = count.view(*reshaper)
        
        return out / count.clamp(min=1e-6)

def robust_scatter_sum(src, index, dim=0, dim_size=None):
    """
    Robust scatter_sum fallback.
    """
    try:
        from torch_scatter import scatter_sum
        return scatter_sum(src, index, dim=dim, dim_size=dim_size)
    except ImportError:
        if dim_size is None:
            dim_size = index.max().item() + 1
        index = index.view(-1)
        out_shape = list(src.shape)
        out_shape[dim] = dim_size
        out = torch.zeros(*out_shape, device=src.device, dtype=src.dtype)
        out.index_add_(dim, index, src)
        return out
