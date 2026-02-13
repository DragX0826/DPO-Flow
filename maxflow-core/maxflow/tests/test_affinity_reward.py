# maxflow/tests/test_affinity_reward.py

import torch
from maxflow.utils.metrics import MultiObjectiveScorer
from torch_geometric.data import Data

def test_integrated_reward():
    print("🧪 Testing Integrated Affinity Reward...")
    scorer = MultiObjectiveScorer()
    
    # 1. Mock Data: Ligand near Protein (Good Affinity)
    # Using single atoms to simplify and avoid internal clashes in this test
    pos_L = torch.tensor([[0.0, 0.0, 0.0]])
    pos_P = torch.tensor([[3.92, 0.0, 0.0]]) # Equilibrium (~ -0.15 kcal/mol)
    
    data_good = Data(pos_L=pos_L, pos_P=pos_P, x_L=torch.randn(1, 167))
    
    # 2. Mock Data: Ligand far from Protein (Poor Affinity)
    pos_P_far = torch.tensor([[100.0, 0.0, 0.0]])
    data_poor = Data(pos_L=pos_L, pos_P=pos_P_far, x_L=torch.randn(1, 167))
    
    # Calculate rewards (mock mol is None, but we check the physics part)
    # weights={'qed': 3.0, 'sa': 1.0, 'clash': -0.5, 'affinity': 0.1}
    reward_good = scorer.calculate_reward(None, pos_L=pos_L, pos_P=pos_P)
    reward_poor = scorer.calculate_reward(None, pos_L=pos_L, pos_P=pos_P_far)
    
    print(f"Reward (Near Protein): {reward_good:.4f}")
    print(f"Reward (Far Protein):  {reward_poor:.4f}")
    
    assert reward_good > reward_poor, "Reward near protein should be higher than far from protein"
    print("✅ Integrated Affinity Reward Test Passed!")

if __name__ == "__main__":
    test_integrated_reward()
