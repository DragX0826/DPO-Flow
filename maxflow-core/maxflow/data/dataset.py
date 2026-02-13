# maxflow/data/dataset.py

import os
import torch
from torch.utils.data import Dataset
from maxflow.data.featurizer import ProteinLigandFeaturizer

class LazyDockingDataset(Dataset):
    """
    Memory-efficient Dataset for Protein-Ligand pairs.
    Only stores file paths and parses data on-the-fly.
    """
    def __init__(self, index_mapping, root_dir=""):
        """
        index_mapping: List of tuples (pdb_rel_path, sdf_rel_path)
        root_dir: Base directory for the structures.
        """
        self.index_mapping = index_mapping
        self.root_dir = root_dir
        self.featurizer = ProteinLigandFeaturizer()

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        entry = self.index_mapping[idx]
        # Robust unpacking: handle cases where mapping has more than 2 elements (metadata)
        pdb_rel = entry[0]
        sdf_rel = entry[1]
        
        pdb_path = os.path.join(self.root_dir, pdb_rel)
        sdf_path = os.path.join(self.root_dir, sdf_rel)
        
        try:
            data = self.featurizer(pdb_path, sdf_path)
            if data is None:
                # Fallback or Skip? 
                # In standard training, we skip during collation.
                return None
            return data
        except Exception as e:
            print(f"Error loading {pdb_rel}: {e}")
            return None

def collate_fn(batch):
    """
    Filter None and use PyG Batch to consolidate.
    Manually creates batch indices for ligand and protein nodes.
    """
    from torch_geometric.data import Batch
    batch = [data for data in batch if data is not None]
    if len(batch) == 0:
        return None
    
    # Standard batching
    out_batch = Batch.from_data_list(batch)
    
    # Manual batch injection for ligand and protein
    x_L_batch = []
    x_P_batch = []
    atom_to_motif_batched = []
    motif_offset = 0
    
    for i, data in enumerate(batch):
        x_L_batch.append(torch.full((data.x_L.size(0),), i, dtype=torch.long))
        x_P_batch.append(torch.full((data.x_P.size(0),), i, dtype=torch.long))
        
        # Motif offsetting for Phase 30
        if hasattr(data, 'atom_to_motif'):
            atom_to_motif_batched.append(data.atom_to_motif + motif_offset)
            motif_offset += data.num_motifs.item()
    
    out_batch.x_L_batch = torch.cat(x_L_batch)
    out_batch.x_P_batch = torch.cat(x_P_batch)
    if atom_to_motif_batched:
        out_batch.atom_to_motif = torch.cat(atom_to_motif_batched)
    
    return out_batch
