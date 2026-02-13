# maxflow/tests/test_phase41_kinetics.py

import torch
from maxflow.utils.physics import PhysicsEngine
from maxflow.models.flow_matching import RectifiedFlow
from maxflow.models.backbone import CrossGVP

def test_escape_barrier_logic():
    print("🧪 Testing Escape Barrier Heuristic...")
    engine = PhysicsEngine()
    
    # 1. Ligand at origin
    pos_L = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    
    # 2. Case: Trapped in a "box" (High barrier in all directions)
    # Surround (0,0,0) with atoms at r=4.0 (near VdW min)
    pos_P_trapped = torch.tensor([
        [4, 0, 0], [-4, 0, 0], [0, 4, 0], [0, -4, 0], [0, 0, 4], [0, 0, -4]
    ], dtype=torch.float32)
    
    barrier_trapped = engine.estimate_escape_barrier(pos_L, pos_P_trapped)
    print(f"  Trapped Barrier: {barrier_trapped.item():.4f}")
    
    # 3. Case: Exposed (Open on one side)
    # Remove one wall at x=-4
    pos_P_exposed = torch.tensor([
        [4, 0, 0], [0, 4, 0], [0, -4, 0], [0, 0, 4], [0, 0, -4]
    ], dtype=torch.float32)
    
    barrier_exposed = engine.estimate_escape_barrier(pos_L, pos_P_exposed)
    print(f"  Exposed Barrier: {barrier_exposed.item():.4f}")
    
    # Exposed barrier should be lower than trapped
    # (Because the open side has a very small barrier to infinity)
    assert barrier_exposed < barrier_trapped

def test_kinetic_guidance_gradient():
    print("\n🧪 Testing Kinetic Guidance Gradients...")
    engine = PhysicsEngine()
    
    # Ligand center at (0,0,0)
    pos_L = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    
    # Protein wall at X=2.0
    pos_P = torch.tensor([[2.0, 0.0, 0.0]])
    
    # Barrier reward (maximize)
    reward = engine.calculate_kinetic_reward(pos_L, pos_P)
    
    # Calculate gradient
    grad = torch.autograd.grad(reward, pos_L)[0]
    print(f"  Kinetic Gradient at (0,0,0): {grad[0].tolist()}")
    
    # To maximize the barrier (distance from the wall at X=2), 
    # the gradient should point NEGATIVE X (away from the wall)
    # Note: If moving +X makes barrier lower (collisions), reward goes down? 
    # Actually, as we move closer to the wall, current energy up, shifted energy up. 
    # Let's check the direction.
    assert grad[0, 0] < 0 

def test_flow_kinetic_integration():
    print("\n🧪 Testing Kinetic Flow Integration...")
    backbone = CrossGVP(node_in_dim=167, hidden_dim=64, num_layers=2)
    model = RectifiedFlow(backbone)
    
    from torch_geometric.data import Data
    data = Data(
        x_L=torch.randn(1, 167),
        pos_L=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        x_P=torch.randn(5, 21),
        pos_P=torch.tensor([[4,0,0], [-4,0,0], [0,4,0], [0,-4,0], [0,0,4]], dtype=torch.float32),
        pocket_center=torch.zeros(1, 3, dtype=torch.float32),
        num_graphs=1,
        x_L_batch=torch.zeros(1, dtype=torch.long),
        x_P_batch=torch.zeros(5, dtype=torch.long)
    )
    
    t = torch.tensor([0.5])
    x_t = data.pos_L.detach().requires_grad_(True)
    
    v_t, v_rot, centers, g_prot = model.get_combined_velocity(t, x_t, data, gamma=1.0)
    
    print(f"  Kinetic-Guided Velocity: {v_t.tolist()}")
    assert v_t.shape == (1, 3)

if __name__ == "__main__":
    test_escape_barrier_logic()
    test_kinetic_guidance_gradient()
    test_flow_kinetic_integration()
