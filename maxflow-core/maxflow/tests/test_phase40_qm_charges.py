# maxflow/tests/test_phase40_qm_charges.py

import torch
import numpy as np
from maxflow.models.flow_matching import RectifiedFlow
from maxflow.models.backbone import CrossGVP
from maxflow.utils.physics import PhysicsEngine

def test_polarizable_charges():
    print("🧪 Testing PAC-GNN Polarizable Charges...")
    engine = PhysicsEngine()
    
    # Ligand atom at (1, 0, 0)
    pos_L = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    q_L_base = torch.tensor([0.0], dtype=torch.float32)
    
    # CASE 1: Neutral Environment
    pos_P_neutral = torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32)
    q_P_neutral = torch.tensor([0.0], dtype=torch.float32)
    q_L_dyn_1 = engine.calculate_polarizable_charges(pos_L, pos_P_neutral, q_L_base, q_P_neutral)
    print(f"  Neutral Env Charge: {q_L_dyn_1.item():.4f}")
    assert abs(q_L_dyn_1.item() - 0.0) < 1e-4

    # CASE 2: Positive Ion nearby
    # Pos ion at (0, 0, 0), Ligand at (1, 0, 0)
    pos_P_ion = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    q_P_ion = torch.tensor([2.0], dtype=torch.float32) # Zn2+ 
    q_L_dyn_2 = engine.calculate_polarizable_charges(pos_L, pos_P_ion, q_L_base, q_P_ion)
    print(f"  Polarized (near Ion) Charge: {q_L_dyn_2.item():.4f}")
    
    # Polarization should shift charge (alpha * sum(q/r2))
    # delta_q = 0.05 * (2.0 / 1.0^2) = 0.1
    assert q_L_dyn_2.item() > 0.05
    assert q_L_dyn_2.item() < 0.15

def test_backbone_charge_head():
    print("\n🧪 Testing Backbone Charge Head Prediction...")
    from torch_geometric.data import Data
    device = torch.device("cpu")
    
    # Mock data
    data = Data(
        x_L=torch.randn(5, 167, device=device),
        pos_L=torch.randn(5, 3, device=device),
        x_P=torch.randn(10, 21, device=device),
        pos_P=torch.randn(10, 3, device=device),
        pocket_center=torch.zeros(1, 3, device=device),
        x_L_batch=torch.zeros(5, dtype=torch.long, device=device),
        x_P_batch=torch.zeros(10, dtype=torch.long, device=device),
        num_graphs=1
    )
    
    backbone = CrossGVP(node_in_dim=167, hidden_dim=64, num_layers=2).to(device)
    t = torch.tensor([0.5], device=device)
    
    v_info, confidence, kl_div = backbone(t, data)
    charge_delta = v_info[3] # 4th element is charge_delta
    
    print(f"  Predicted Charge Delta Shape: {charge_delta.shape}")
    print(f"  Mean Delta: {charge_delta.mean().item():.4f}")
    assert charge_delta.shape == (5, 1)
    assert torch.all(charge_delta >= -1.0) and torch.all(charge_delta <= 1.0)

def test_qm_guidance_flow():
    print("\n🧪 Testing QM-Corrected Guidance in RectifiedFlow...")
    backbone = CrossGVP(node_in_dim=167, hidden_dim=64, num_layers=2)
    model = RectifiedFlow(backbone)
    
    from torch_geometric.data import Data
    data = Data(
        x_L=torch.randn(2, 167),
        pos_L=torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        x_P=torch.randn(2, 21),
        pos_P=torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        q_P=torch.tensor([1.0, -1.0]), # Dipole protein
        pocket_center=torch.zeros(1, 3),
        num_graphs=1,
        x_L_batch=torch.zeros(2, dtype=torch.long),
        x_P_batch=torch.zeros(2, dtype=torch.long)
    )
    
    t = torch.tensor([0.5])
    # x_t should require grad for guidance calculation
    x_t = data.pos_L.detach().requires_grad_(True)
    
    v_t, v_rot, centers, g_prot = model.get_combined_velocity(t, x_t, data, gamma=1.0)
    
    # We verify it runs without error and v_t has some magnitude
    print(f"  Guidance Velocity Magnitude: {torch.norm(v_t).item():.4f}")
    assert v_t.shape == (2, 3)

if __name__ == "__main__":
    test_polarizable_charges()
    test_backbone_charge_head()
    test_qm_guidance_flow()
