# max_flow/tests/test_phase42_ensemble.py

import torch
import numpy as np
from max_flow.models.flow_matching import RectifiedFlow
from max_flow.models.backbone import CrossGVP
from max_flow.utils.physics import PhysicsEngine

def test_ensemble_energy_logic():
    print("🧪 Testing Ensemble-Averaged Energy...")
    engine = PhysicsEngine()
    
    # 1. Ligand at (0,0,0)
    pos_L = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    
    # 2. Ensemble of 2 protein states
    # State 1: Atom at (4,0,0) - Weight 0.7
    # State 2: Atom at (5,0,0) - Weight 0.3
    ensemble = [
        {'pos_P': torch.tensor([[4.0, 0.0, 0.0]]), 'q_P': torch.tensor([1.0]), 'weight': 0.7},
        {'pos_P': torch.tensor([[5.0, 0.0, 0.0]]), 'q_P': torch.tensor([1.0]), 'weight': 0.3}
    ]
    
    e_ensemble = engine.calculate_ensemble_interaction_energy(pos_L, ensemble)
    
    # Manually calculate expected
    e1 = engine.calculate_interaction_energy(pos_L, ensemble[0]['pos_P'], q_P=ensemble[0]['q_P'])
    e2 = engine.calculate_interaction_energy(pos_L, ensemble[1]['pos_P'], q_P=ensemble[1]['q_P'])
    expected = 0.7 * e1 + 0.3 * e2
    
    print(f"  Ensemble Energy: {e_ensemble.item():.4f}")
    assert abs(e_ensemble.item() - expected.item()) < 1e-4

def test_backbone_ensemble_forward():
    print("\n🧪 Testing Backbone Ensemble-Aware Forward...")
    from torch_geometric.data import Data
    device = torch.device("cpu")
    
    # 1. Mock ensemble data
    ensemble_P = [
        {'pos_P': torch.randn(10, 3), 'x_P': torch.randn(10, 21)},
        {'pos_P': torch.randn(10, 3), 'x_P': torch.randn(10, 21)}
    ]
    
    data = Data(
        x_L=torch.randn(5, 167),
        pos_L=torch.randn(5, 3),
        pocket_center=torch.zeros(1, 3),
        ensemble_P=ensemble_P,
        num_graphs=1,
        x_L_batch=torch.zeros(5, dtype=torch.long)
    )
    # Also set default pos_P for metadata/time logic
    data.pos_P = ensemble_P[0]['pos_P']
    data.x_P = ensemble_P[0]['x_P']
    data.x_P_batch = torch.zeros(10, dtype=torch.long)
    
    backbone = CrossGVP(node_in_dim=167, hidden_dim=64, num_layers=2)
    t = torch.tensor([0.5])
    
    # 2. Run Forward
    v_info, confidence, kl_div = backbone(t, data)
    
    # Unpack based on P40 return signature: ((v, rot), water, admet, charge)
    (v_v_info, p_water, admet_pred, charge_delta) = v_info
    
    print(f"  Forward successful with Ensemble.")
    assert v_v_info.shape == (5, 3) # atom level: 5 atoms, 3 dimensions

def test_flow_ensemble_guidance():
    print("\n🧪 Testing Ensemble-Weighted Guidance in Flow...")
    backbone = CrossGVP(node_in_dim=167, hidden_dim=64, num_layers=2)
    model = RectifiedFlow(backbone)
    
    from torch_geometric.data import Data
    ensemble_P = [
        {'pos_P': torch.tensor([[4.0, 0, 0]]), 'q_P': torch.tensor([2.0]), 'weight': 0.9},
        {'pos_P': torch.tensor([[-4.0, 0, 0]]), 'q_P': torch.tensor([-2.0]), 'weight': 0.1}
    ]
    
    data = Data(
        x_L=torch.randn(1, 167),
        pos_L=torch.tensor([[0.0, 0.0, 0.0]]),
        pocket_center=torch.zeros(1, 3),
        ensemble_P=ensemble_P,
        num_graphs=1,
        x_L_batch=torch.zeros(1, dtype=torch.long)
    )
    data.pos_P = ensemble_P[0]['pos_P']
    data.x_P = torch.randn(1, 21)
    data.x_P_batch = torch.zeros(1, dtype=torch.long)
    
    t = torch.tensor([0.5])
    x_t = data.pos_L.detach().requires_grad_(True)
    
    # In P42, gamma=1.0 will trigger ensemble energy calculation
    v_t, v_rot, centers, g_prot = model.get_combined_velocity(t, x_t, data, gamma=1.0)
    
    print(f"  Ensemble Guidance Magnitude: {torch.norm(v_t).item():.4f}")
    assert v_t.shape == (1, 3)

if __name__ == "__main__":
    test_ensemble_energy_logic()
    test_backbone_ensemble_forward()
    test_flow_ensemble_guidance()
