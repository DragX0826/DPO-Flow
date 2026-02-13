# maxflow/tests/test_physics_engine.py

import torch
from maxflow.utils.physics import compute_vdw_energy, compute_electrostatic_energy, calculate_affinity_reward

def test_physics_kernels():
    print("🧪 Testing Physics Kernels...")
    
    # 1. Mock Data: C-C interaction
    # Equilibrium distance for sigma=3.5 is approx 3.92A (2^(1/6) * sigma)
    pos_L = torch.tensor([[0.0, 0.0, 0.0]])
    pos_P = torch.tensor([[3.92, 0.0, 0.0]])
    
    e_vdw = compute_vdw_energy(pos_L, pos_P, epsilon=0.15, sigma=3.5)
    print(f"vdW Energy at equilibrium: {e_vdw.item():.4f} kcal/mol (Expected ~ -0.15)")
    
    # 2. Mock Data: Opposite charges
    q_ligand = torch.tensor([0.2])
    q_pocket = torch.tensor([-0.2])
    pos_P_charge = torch.tensor([[5.0, 0.0, 0.0]]) # 5A distance
    
    e_elec = compute_electrostatic_energy(pos_L, pos_P_charge, q_ligand, q_pocket, dielectric=1.0)
    # E = 332 * (0.2 * -0.2) / 5 = 332 * -0.04 / 5 = -13.28 / 5 = -2.656
    print(f"Electrostatic Energy: {e_elec.item():.4f} kcal/mol (Expected ~ -2.656)")
    
    # 3. Overall Reward
    reward = calculate_affinity_reward(pos_L, pos_P)
    print(f"Affinity Reward: {reward.item():.4f}")
    
    assert e_vdw < 0, "vdW at equilibrium should be negative"
    assert e_elec < 0, "Opposite charges should have negative energy"
    print("✅ Physics Kernel Test Passed!")

if __name__ == "__main__":
    test_physics_kernels()
