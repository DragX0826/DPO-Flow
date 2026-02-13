# maxflow/tests/test_multitarget_physics.py

import torch
from maxflow.utils.physics import PhysicsEngine

def test_multitarget_energy():
    print("🧪 Testing Multitarget Energy (PROTAC Ternary Complex)...")
    engine = PhysicsEngine()
    
    pos_L = torch.randn(10, 3)
    # targets is a list of (pos_P, q_P) tuples
    target1 = (torch.randn(20, 3), torch.randn(20))
    target2 = (torch.randn(20, 3), torch.randn(20))
    
    energy = engine.calculate_multitarget_energy(pos_L, [target1, target2])
    
    print(f"  Ternary Complex Energy: {energy.item():.4f}")
    assert energy.item() != 0.0

def test_selectivity_penalty():
    print("\n🧪 Testing Selectivity Penalty (Mutant vs WT)...")
    engine = PhysicsEngine()
    
    pos_L = torch.randn(10, 3)
    pos_mutant = torch.randn(20, 3)
    pos_wildtype = pos_mutant + 0.5  # Slightly different
    
    penalty = engine.calculate_selectivity_penalty(pos_L, pos_mutant, pos_wildtype)
    
    print(f"  Selectivity Penalty: {penalty.item():.4f}")
    assert penalty.item() >= 0.0

def test_covalent_potential():
    print("\n🧪 Testing Covalent Potential (Warhead Geometry)...")
    engine = PhysicsEngine()
    
    pos_L = torch.zeros(5, 3, requires_grad=True)
    pos_L.data[0] = torch.tensor([2.5, 0.0, 0.0])  # Warhead at 2.5A
    pos_P = torch.zeros(20, 3)  # Target at origin
    covalent_indices = torch.tensor([[0, 0]], dtype=torch.long)
    
    e_cov = engine.calculate_covalent_potential(pos_L, pos_P, covalent_indices)
    
    print(f"  Covalent Potential (dist=2.5A): {e_cov.item():.4f}")
    assert e_cov.item() > 0.0  # Non-zero energy

def test_admet_scores():
    print("\n🧪 Testing ADMET Scores (Compactness Proxy)...")
    engine = PhysicsEngine()
    
    pos_L = torch.randn(10, 3)
    x_L = torch.randn(10, 167)
    
    score = engine.calculate_admet_scores(pos_L, x_L)
    
    print(f"  ADMET Compactness Score: {score.item():.4f}")
    assert score.item() >= 0.0

if __name__ == "__main__":
    test_multitarget_energy()
    test_selectivity_penalty()
    test_covalent_potential()
    test_admet_scores()
