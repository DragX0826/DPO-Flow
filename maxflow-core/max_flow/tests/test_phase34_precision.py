# max_flow/tests/test_phase34_precision.py

import torch
import numpy as np
from max_flow.models.flow_matching import RectifiedFlow
from max_flow.models.backbone import CrossGVP
from max_flow.utils.physics import PhysicsEngine

def test_rigid_body_rotation():
    print("🧪 Testing SE(3) Rigid Body Rotation (Rodrigues)...")
    backbone = CrossGVP(node_in_dim=167, hidden_dim=64, num_layers=2)
    model = RectifiedFlow(backbone)
    
    # Create two atoms in one motif
    x_t = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32)
    v_trans = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    v_rot = torch.tensor([[0.0, 0.0, torch.pi/2]], dtype=torch.float32) # 90 deg/s around Z
    centroids = torch.tensor([[1.5, 0.0, 0.0]], dtype=torch.float32)
    atom_to_motif = torch.tensor([0, 0], dtype=torch.long)
    dt = 1.0
    
    x_new = model.rigid_body_step(x_t, v_trans, v_rot, centroids, dt, atom_to_motif)
    
    # Relative vectors: [-0.5, 0, 0] and [0.5, 0, 0]
    # Rotate 90 deg around Z: [0, -0.5, 0] and [0, 0.5, 0]
    # New positions: [1.5, -0.5, 0] and [1.5, 0.5, 0]
    expected = torch.tensor([[1.5, -0.5, 0.0], [1.5, 0.5, 0.0]], dtype=torch.float32)
    
    error = torch.norm(x_new - expected)
    print(f"  Rotation Error: {error.item():.6f}")
    assert error < 1e-5

def test_surface_normal_estimation():
    print("\n🧪 Testing Surface Normal Estimation...")
    engine = PhysicsEngine()
    
    # Create a simple flat surface in XY plane at Z=0
    # Add many points to ensure centroid is below
    pos_P = torch.tensor([
        [0.0, 0.0, 0.0], # Target atom
        [1.0, 0.0, -0.1], 
        [-1.0, 0.0, -0.1], 
        [0.0, 1.0, -0.1], 
        [0.0, -1.0, -0.1]
    ], dtype=torch.float32)
    
    normals = engine.estimate_surface_normals(pos_P)
    target_normal = normals[0]
    
    # Centroid of neighbors is at (0, 0, -0.1)
    # Target is at (0, 0, 0)
    # Normal = (0, 0, 0) - (0, 0, -0.1) = (0, 0, 0.1) -> unit: (0, 0, 1)
    print(f"  Estimated Normal: {target_normal.numpy()}")
    assert target_normal[2] > 0.9 # Should point along +Z

def test_heun_integration():
    print("\n🧪 Testing 2nd-order Heun Integration Workflow...")
    from torch_geometric.data import Data
    data = Data(
        x_L=torch.randn(5, 167),
        pos_L=torch.randn(5, 3),
        x_P=torch.randn(10, 21),
        pos_P=torch.randn(10, 3),
        pocket_center=torch.zeros(1, 3),
        atom_to_motif=torch.zeros(5, dtype=torch.long)
    )
    # Mock batch attributes manually to bypass Batch.from_data_list magic in test
    data.num_graphs = 1
    data.x_L_batch = torch.zeros(5, dtype=torch.long)
    data.x_P_batch = torch.zeros(10, dtype=torch.long)
    
    backbone = CrossGVP(node_in_dim=167, hidden_dim=64, num_layers=2)
    model = RectifiedFlow(backbone)
    
    # Run a short sample loop with 2 steps
    # Enable grads for physics/guidance calculation (Phase 38 needs this)
    data.pos_L = data.pos_L.detach().requires_grad_(True)
    data.pos_P = data.pos_P.detach().requires_grad_(True)
    
    try:
        x_final, traj = model.sample(data, steps=2, gamma=1.0)
        print(f"  Heun Sampling Success. Trajectory length: {len(traj)}")
        assert len(traj) == 3 # t=0, 0.5, 1.0
    except Exception as e:
        print(f"  Heun Sampling Failed: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    test_rigid_body_rotation()
    test_surface_normal_estimation()
    test_heun_integration()
