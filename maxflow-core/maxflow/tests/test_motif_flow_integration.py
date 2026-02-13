# maxflow/tests/test_motif_flow_integration.py

import torch
import os
from maxflow.data.featurizer import ProteinLigandFeaturizer
from maxflow.models.backbone import CrossGVP
from maxflow.models.flow_matching import RectifiedFlow
from torch_geometric.data import Batch

def test_motif_flow_pipeline():
    print("🧪 Testing Motif Flow Integration...")
    
    # 1. Setup Data (Use mock or existing if available)
    # For a smoke test, we'll manually create a Data object with motif info
    num_atoms = 10
    num_protein_residues = 20
    
    x_L = torch.randn(num_atoms, 167)
    pos_L = torch.randn(num_atoms, 3)
    edge_index_L = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    
    x_P = torch.randn(num_protein_residues, 21)
    pos_P = torch.randn(num_protein_residues, 3)
    
    atom_to_motif = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long) # 3 motifs
    num_motifs = torch.tensor([3], dtype=torch.long)
    
    from torch_geometric.data import Data
    data = Data(
        x_L=x_L, pos_L=pos_L, edge_index_L=edge_index_L,
        x_P=x_P, pos_P=pos_P,
        pocket_center=torch.zeros(1, 3),
        atom_to_motif=atom_to_motif,
        num_motifs=num_motifs
    )
    
    # Batch it
    batch = Batch.from_data_list([data, data])
    # Manual offsetting (as in our updated collate_fn)
    batch.x_L_batch = torch.cat([torch.zeros(num_atoms, dtype=torch.long), torch.ones(num_atoms, dtype=torch.long)])
    batch.x_P_batch = torch.cat([torch.zeros(num_protein_residues, dtype=torch.long), torch.ones(num_protein_residues, dtype=torch.long)])
    batch.atom_to_motif = torch.cat([atom_to_motif, atom_to_motif + 3])
    
    # 2. Setup Model
    backbone = CrossGVP(node_in_dim=167, hidden_dim=64, num_layers=2)
    flow_model = RectifiedFlow(backbone)
    
    # 3. Test Forward/Loss
    print("  Running Loss Calculation...")
    loss = flow_model.loss(batch)
    print(f"  Loss: {loss.item():.4f}")
    assert not torch.isnan(loss)
    
    # 4. Test Sample
    # Note: Sample is not yet fully updated for Motif-level sampling in flow_matching.py
    # (It still uses atom-level Euler step, but it should be compatible)
    print("  Running Sample step...")
    x_sampled, traj = flow_model.sample(batch, steps=2)
    print(f"  Sampled Pos Shape: {x_sampled.shape}")
    assert x_sampled.shape == torch.Size([20, 3])
    
    print("✅ Motif Flow Integration Test Passed!")

if __name__ == "__main__":
    test_motif_flow_pipeline()
