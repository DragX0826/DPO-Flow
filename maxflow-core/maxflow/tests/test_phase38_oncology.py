# maxflow/tests/test_phase38_oncology.py

import torch
from maxflow.models.max_rl import MaxFlow
from maxflow.utils.physics import PhysicsEngine
from torch_geometric.data import Data, Batch

def test_covalent_geometry_guidance():
    print("🧪 Testing Covalent Geometry Guidance...")
    model = MaxFlow()
    physics = PhysicsEngine()
    
    # Mock data with covalent indices (Warhead -> Target)
    num_atoms = 10
    pos_L = torch.randn(num_atoms, 3, requires_grad=True)
    pos_P = torch.zeros(20, 3) # Target residue at origin
    covalent_indices = torch.tensor([[0, 12]], dtype=torch.long) # Atom 0 to Protein index 12
    
    # Initial distance (Target attraction)
    pos_L = torch.zeros(num_atoms, 3, requires_grad=True)
    pos_L.data[0] = torch.tensor([3.0, 0.0, 0.0])
    pos_P = torch.zeros(20, 3) # Target residue 12 at origin
    dist_init = torch.norm(pos_L[0] - pos_P[12])
    
    # Calculate Covalent Potential
    e_cov = physics.calculate_covalent_potential(pos_L, pos_P, covalent_indices)
    grad = torch.autograd.grad(e_cov, pos_L)[0]
    direction = pos_P[12] - pos_L[0]
    
    # Calculate Force Direction
    force = -grad[0]
    cosine = torch.cosine_similarity(force, direction, dim=0)
    
    print(f"  Distance: {dist_init.item():.4f} | Force Direction Cosine: {cosine.item():.4f}")
    assert cosine.item() > 0.9 # Attractive

def test_admet_head_integration():
    print("\n🧪 Testing ADMET Head Integration...")
    model = MaxFlow()
    
    data = Data(
        x_L=torch.randn(5, 167),
        pos_L=torch.randn(5, 3),
        x_P=torch.randn(10, 21),
        pos_P=torch.randn(10, 3),
        pocket_center=torch.zeros(1, 3),
        num_graphs=1,
        x_L_batch=torch.zeros(5, dtype=torch.long)
    )
    batch = Batch.from_data_list([data])
    
    # Forward pass
    v_info, confidence, kl_div = model.flow.backbone(torch.tensor([0.5]), batch)
    
    # Unpack based on new Phase 38 signature
    # ((v_trans, v_rot), p_water, admet_pred) or (v_trans, p_water, admet_pred)
    if isinstance(v_info[0], tuple):
        _, _, admet_pred = v_info
    else:
        _, _, admet_pred = v_info
        
    print(f"  Predicted ADMET [LogP, Solubility]: {admet_pred.detach().cpu().numpy()}")
    assert admet_pred.shape == (1, 2)

if __name__ == "__main__":
    test_covalent_geometry_guidance()
    test_admet_head_integration()
