# maxflow/utils/geometry.py

import torch

def radius_graph(x, r, batch=None, loop=False, max_num_neighbors=32, batch_x=None, batch_y=None, chunk_size=1024):
    """
    Computes radius graph using pure PyTorch with chunking to avoid OOM.
    
    Args:
        x (Tensor or Tuple): Node coordinates (N, 3) or (source, target).
        r (float): Radius threshold.
        batch (LongTensor, optional): Batch vector.
        loop (bool): If True, include self-loops.
        chunk_size (int): Number of nodes to process at once for the distance matrix.
    """
    if isinstance(x, (tuple, list)):
        x_source, x_target = x
        batch_source = batch_x if batch_x is not None else torch.zeros(x_source.size(0), dtype=torch.long, device=x_source.device)
        batch_target = batch_y if batch_y is not None else torch.zeros(x_target.size(0), dtype=torch.long, device=x_target.device)
    else:
        x_source = x_target = x
        batch_source = batch_target = batch if batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

    device = x_source.device
    target_indices = []
    source_indices = []
    
    # Process target nodes in chunks to keep distance matrix (chunk_size, N_source) manageable
    for i in range(0, x_target.size(0), chunk_size):
        end = min(i + chunk_size, x_target.size(0))
        x_target_chunk = x_target[i:end]
        batch_target_chunk = batch_target[i:end]
        
        # d: (chunk_size, N_source)
        dist = torch.cdist(x_target_chunk, x_source)
        
        # Mask by radius and batch
        mask = (dist < r) & (batch_target_chunk.unsqueeze(1) == batch_source.unsqueeze(0))
        
        # Self-loop handling
        if not loop and x_source is x_target:
            # Mask out diag: index j in chunk corresponds to index i+j in source
            n_chunk = end - i
            diag_idx = torch.arange(n_chunk, device=device)
            source_idx_in_chunk = i + diag_idx
            # Only mask if source_idx is within x_source bounds
            valid_diag = source_idx_in_chunk < x_source.size(0)
            mask[diag_idx[valid_diag], source_idx_in_chunk[valid_diag]] = False

        # Get indices and add offset
        c_target_idx, c_source_idx = mask.nonzero(as_tuple=True)
        target_indices.append(c_target_idx + i)
        source_indices.append(c_source_idx)
        
    if not target_indices:
        return torch.empty((2, 0), dtype=torch.long, device=device)
        
    edge_index = torch.stack([torch.cat(source_indices), torch.cat(target_indices)], dim=0)
    
    # Optional: limit neighbors per node (max_num_neighbors)
    # This is more complex to do efficiently without torch_cluster.
    # For now, we return all within radius.
    
    return edge_index

def knn_graph(x, k, batch=None, loop=False, flow='source_to_target'):
    """
    Computes k-nearest neighbor graph using pure PyTorch.
    
    Args:
        x (Tensor): Node coordinates (N, 3).
        k (int): Number of neighbors.
        batch (LongTensor, optional): Batch vector.
        loop (bool): If True, include self-loops.
    """
    device = x.device
    num_nodes = x.size(0)
    if num_nodes == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
        
    # Compute full distance matrix (N, N)
    # For large N, this might need chunking, but usually L+P nodes < 5000.
    dist = torch.cdist(x, x)
    
    # Batch mask: only connect within same batch
    if batch is not None:
        mask = (batch.unsqueeze(1) == batch.unsqueeze(0))
        dist = dist.masked_fill(~mask, float('inf'))
        
    if not loop:
        # Fill diagonal with inf to skip self-loops
        indices = torch.arange(num_nodes, device=device)
        dist[indices, indices] = float('inf')
        
    # Get top-k smallest distances
    # values, col = torch.topk(dist, k=k, dim=1, largest=False)
    # Using sort is safer if k > num_nodes
    k = min(k, num_nodes - (0 if loop else 1))
    if k <= 0:
         return torch.empty((2, 0), dtype=torch.long, device=device)
         
    _, col = torch.topk(dist, k=k, dim=1, largest=False)
    
    row = torch.arange(num_nodes, device=device).unsqueeze(1).expand_as(col)
    
    edge_index = torch.stack([col.reshape(-1), row.reshape(-1)], dim=0)
    return edge_index

def radius(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=32):
    """
    Wrapper for bipartite radius graph matching torch_cluster signature.
    x: source (Protein)
    y: target (Ligand)
    """
    return radius_graph((x, y), r, batch_x=batch_x, batch_y=batch_y, max_num_neighbors=max_num_neighbors, loop=False)

def scatter_add(src, index, dim=0, dim_size=None):
    """
    Pure torch implementation of scatter_add.
    Fallback for torch_scatter.scatter.
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0
        
    shape = list(src.shape)
    shape[dim] = dim_size
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    
    # Broadcast index if necessary
    if src.dim() > 1:
        # Construct views to expand index to match src
        view = [1] * src.dim()
        view[dim] = -1
        index_expanded = index.view(view).expand_as(src)
    else:
        index_expanded = index
        
    return out.scatter_add_(dim, index_expanded, src)
