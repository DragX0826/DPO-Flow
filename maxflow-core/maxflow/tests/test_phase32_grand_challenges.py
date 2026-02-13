# maxflow/tests/test_phase32_grand_challenges.py

import torch
import numpy as np
import pytest
from maxflow.data.featurizer import ProteinLigandFeaturizer
from maxflow.models.backbone import CrossGVP
from maxflow.models.flow_matching import RectifiedFlow
from maxflow.utils.physics import PhysicsEngine
from torch_geometric.data import Data

def test_metal_detection_and_coordination():
    # 1. Mock a protein residue + Zn ion
    # We'll manually create a Data object to simulate featurizer output
    pos_L = torch.randn(5, 3)
    pos_P = torch.randn(10, 3)
    pos_metals = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32) # Zn at origin
    
    # 2. Physics Check
    engine = PhysicsEngine()
    
    # Ligand atom at 2.1A from Zn (ideal coordination)
    pos_L_ideal = torch.tensor([[2.1, 0.0, 0.0]], dtype=torch.float32)
    energy_ideal = engine.calculate_interaction_energy(pos_L_ideal, pos_P, pos_metals=pos_metals)
    
    # Ligand atom at 1.0A (clash/too close)
    pos_L_close = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    energy_close = engine.calculate_interaction_energy(pos_L_close, pos_P, pos_metals=pos_metals)
    
    print(f"Ideal Coordination (2.1A) Energy: {energy_ideal.item():.4f}")
    print(f"Close Coordination (1.0A) Energy: {energy_close.item():.4f}")
    
    # Energy at 1.0A should be much higher due to 10-12 well repulsion
    assert energy_close > energy_ideal

def test_water_head_sampling():
    # 1. Setup Model
    hidden_dim = 64
    backbone = CrossGVP(node_in_dim=167, hidden_dim=hidden_dim, num_layers=2)
    model = RectifiedFlow(backbone)
    
    # 2. Mock Batch with Phase 32 features
    from torch_geometric.data import Batch
    data = Data(
        x_L=torch.randn(10, 167),
        pos_L=torch.randn(10, 3),
        x_P=torch.randn(20, 21),
        pos_P=torch.randn(20, 3),
        pos_metals=torch.zeros(1, 3),
        pocket_center=torch.zeros(1, 3),
        atom_to_motif=torch.zeros(10, dtype=torch.long)
    )
    batch = Batch.from_data_list([data])
    
    # 3. Test Sampling with Guidance
    # gamma > 0 triggers confidence and hydration guidance
    x_gen, traj = model.sample(batch, steps=5, gamma=1.0)
    
    assert x_gen.shape == (10, 3)
    assert len(traj) == 6 # initial + 5 steps
    print("Phase 32 Sampling integration successful.")

if __name__ == "__main__":
    test_metal_detection_and_coordination()
    test_water_head_sampling()
