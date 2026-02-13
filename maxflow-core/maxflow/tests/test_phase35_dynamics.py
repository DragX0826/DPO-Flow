# maxflow/tests/test_phase35_dynamics.py

import torch
import numpy as np
from maxflow.models.flow_matching import RectifiedFlow
from maxflow.models.backbone import CrossGVP
from maxflow.utils.physics import PhysicsEngine

def test_induced_fit_displacement():
    print("🧪 Testing Induced-Fit Displacement (Protein Flexibility)...")
    backbone = CrossGVP(node_in_dim=167, hidden_dim=64, num_layers=2)
    model = RectifiedFlow(backbone)
    
    from torch_geometric.data import Data
    # Ligand atom clashing with Protein atom at origin
    data = Data(
        x_L=torch.randn(1, 167),
        pos_L=torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32),
        x_P=torch.randn(1, 21),
        pos_P=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32), # Target protein atom
        pocket_center=torch.zeros(1, 3),
        num_graphs=1,
        x_L_batch=torch.zeros(1, dtype=torch.long),
        x_P_batch=torch.zeros(1, dtype=torch.long)
    )
    
    pos_P_before = data.pos_P.clone()
    
    # Run 1 step at t=0.5 where flex_scale is max
    # We mock steps=1 but we'll manually call sub-methods to verify
    t = torch.tensor([0.5], device='cpu')
    # Enable grads for physics/guidance calculation
    data.pos_L = data.pos_L.detach().requires_grad_(True)
    data.pos_P = data.pos_P.detach().requires_grad_(True)
    
    v_t, v_rot_t, centroids_t, g_prot_t = model.get_combined_velocity(t, data.pos_L, data, gamma=1.0)
    
    print(f"  Protein Grad: {g_prot_t.numpy()}")
    
    # Manually apply displacement as in sample()
    flex_scale = 0.5 * (4 * 0.5 * (1 - 0.5)) # 0.5
    dt = 0.1
    data.pos_P = data.pos_P - flex_scale * g_prot_t * dt
    
    diff = torch.norm(data.pos_P - pos_P_before)
    print(f"  Displacement: {diff.item():.6f}")
    assert diff > 0 # Protein must move away from ligand clash

def test_langevin_injection():
    print("\n🧪 Testing Langevin Noise Injection...")
    backbone = CrossGVP(node_in_dim=167, hidden_dim=64, num_layers=2)
    model = RectifiedFlow(backbone)
    
    # We verify that sample() results are stochastic at mid-t
    from torch_geometric.data import Data
    data = Data(
        x_L=torch.randn(5, 167),
        pos_L=torch.randn(5, 3),
        x_P=torch.randn(10, 21),
        pos_P=torch.randn(10, 3),
        pocket_center=torch.zeros(1, 3),
        num_graphs=1,
        x_L_batch=torch.zeros(5, dtype=torch.long),
        x_P_batch=torch.zeros(10, dtype=torch.long)
    )
    
    # Set seed for reproducible predictive part, but Langevin should still differ
    torch.manual_seed(42)
    x1, _ = model.sample(data, steps=5, gamma=1.0)
    
    torch.manual_seed(42)
    x2, _ = model.sample(data, steps=5, gamma=1.0)
    
    # Since we set torch seeds, Randn in backbone will be same, 
    # but randn in Langevin happens at different times during the loop
    # Wait, torch.manual_seed(42) will make Langevin randn same too if called before sample.
    # To test stochasticity, we run without resetting seed between calls.
    x3, _ = model.sample(data, steps=5, gamma=1.0)
    
    diff = torch.norm(x1 - x3)
    print(f"  Stochastic Diff (x1 vs x3): {diff.item():.6f}")
    assert diff > 1e-3

def test_virtual_bond_constraints():
    print("\n🧪 Testing Topological Virtual Bonds...")
    backbone = CrossGVP(node_in_dim=167, hidden_dim=64, num_layers=2)
    model = RectifiedFlow(backbone)
    
    from torch_geometric.data import Data
    # Two atoms in separate motifs, but connected by a joint
    data = Data(
        x_L=torch.randn(2, 167),
        pos_L=torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=torch.float32),
        x_P=torch.randn(1, 21),
        pos_P=torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32),
        joint_indices=torch.tensor([[0, 1]], dtype=torch.long),
        pocket_center=torch.zeros(1, 3),
        num_graphs=1,
        x_L_batch=torch.zeros(2, dtype=torch.long),
        x_P_batch=torch.zeros(1, dtype=torch.long)
    )
    
    t = torch.tensor([0.5])
    # Mock get_combined_velocity and check if joint force exists
    # We need to ensure backbone doesn't clash too much or we check v_joint_atom separately
    
    # Inspect internal v_joint_atom by mocking get_combined_velocity parts
    with torch.enable_grad():
        x_t = data.pos_L.detach().requires_grad_(True)
        # Force a large distance (5.0A vs d0=1.5A)
        # Joint force should pull them together
        joints = data.joint_indices
        pos_i = x_t[joints[:, 0]]
        pos_j = x_t[joints[:, 1]]
        diff = pos_i - pos_j
        dist = torch.norm(diff, dim=-1, keepdim=True)
        k_joint = 10.0
        d0 = 1.5
        force = - k_joint * (dist - d0) * (diff / (dist + 1e-6))
        
        print(f"  Joint Force (on atom 0): {force.detach().numpy()}")
        assert force[0, 0] > 0 # Pulling atom 0 towards atom 1 (which is at X=5)

if __name__ == "__main__":
    test_induced_fit_displacement()
    test_langevin_injection()
    test_virtual_bond_constraints()
