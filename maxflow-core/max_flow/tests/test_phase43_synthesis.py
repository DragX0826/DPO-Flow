# max_flow/tests/test_phase43_synthesis.py

import torch
from max_flow.utils.physics import PhysicsEngine
from max_flow.models.flow_matching import RectifiedFlow
from max_flow.models.backbone import CrossGVP

def test_synthetic_reward_logic():
    print("🧪 Testing Synthetic Reward Logic...")
    engine = PhysicsEngine()
    
    # Standard 6-atom ring-like setup (Aspirin-ish)
    pos_L_drug = torch.tensor([
        [0, 0, 0], [1.4, 0, 0], [2.1, 1.2, 0],
        [1.4, 2.4, 0], [0, 2.4, 0], [-0.7, 1.2, 0]
    ], dtype=torch.float32)
    
    # Exploded / Messy setup (High Rg penalty)
    pos_L_messy = torch.tensor([
        [0, 0, 0], [10, 0, 0], [20, 0, 0],
        [0, 10, 0], [0, 20, 0], [10, 10, 10]
    ], dtype=torch.float32)
    
    x_L = torch.randn(6, 167)
    
    # 1. Test Rg penalty
    r_drug = engine.calculate_synthetic_reward(pos_L_drug, x_L, None)
    r_messy = engine.calculate_synthetic_reward(pos_L_messy, x_L, None)
    
    print(f"  Drug SA Reward: {r_drug.item():.4f}")
    print(f"  Messy SA Reward: {r_messy.item():.4f}")
    
    assert r_drug > r_messy # Drug should be much more rewarded (less penalty)

    # 2. Test Motif Connectivity
    # atom_to_motif: [0, 0, 0, 1, 1, 1] - two 3-atom motifs
    atom_to_motif = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    
    # Case A: Motifs are close (Connected)
    pos_L_conn = torch.tensor([
        [0,0,0], [1,0,0], [0,1,0], # Motif 0
        [1.5,0,0], [2.5,0,0], [1.5,1,0] # Motif 1 (Center at ~1.8 from Motif 0)
    ], dtype=torch.float32, requires_grad=True)
    
    # Case B: Motifs are far apart (Fragmented)
    pos_L_frag = torch.tensor([
        [0,0,0], [1,0,0], [0,1,0],
        [10,10,0], [11,10,0], [10,11,0]
    ], dtype=torch.float32, requires_grad=True)
    
    r_conn = engine.calculate_synthetic_reward(pos_L_conn, x_L, atom_to_motif)
    r_frag = engine.calculate_synthetic_reward(pos_L_frag, x_L, atom_to_motif)
    
    print(f"  Connected Reward: {r_conn.item():.4f}")
    print(f"  Fragmented Reward: {r_frag.item():.4f}")
    
    assert r_conn > r_frag

def test_synthesis_guidance_gradient():
    print("\n🧪 Testing Synthesis Guidance Gradients...")
    engine = PhysicsEngine()
    
    # Create local tensor to be SURE it's a leaf
    # Use 1.6A separation (near the 1.5A target)
    pos_L_leaf = torch.tensor([
        [0.0, 0.0, 0.0], [1.6, 0.0, 0.0]
    ], dtype=torch.float32, requires_grad=True)
    
    atom_to_motif = torch.tensor([0, 1])
    x_L = torch.randn(2, 167)
    
    reward = engine.calculate_synthetic_reward(pos_L_leaf, x_L, atom_to_motif)
    grad = torch.autograd.grad(reward, pos_L_leaf)[0]
    
    print(f"  Gradient at Separation 5.0A: {grad.tolist()}")
    
    # To maximize reward (bring motifs to ~1.5A), 
    # grad[0] should be positive (move toward +X)
    # grad[1] should be negative (move toward -X)
    assert grad[0, 0] > 0
    assert grad[1, 0] < 0

if __name__ == "__main__":
    test_synthetic_reward_logic()
    test_synthesis_guidance_gradient()
