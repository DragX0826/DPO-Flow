import torch
from maxflow.models.backbone import CrossGVP
from maxflow.models.flow_matching import RectifiedFlow
from torch_geometric.data import Data, Batch

def test_guidance_gradients():
    print("Verifying Confidence Gradients for SOTA 2.0...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Model
    backbone = CrossGVP(node_in_dim=161, hidden_dim=64).to(device)
    rf = RectifiedFlow(backbone)
    
    # 2. Setup Mock Data
    num_nodes = 10
    data = Data(
        x_L=torch.randn(num_nodes, 161),
        pos_L=torch.randn(num_nodes, 3).requires_grad_(True),
        x_P=torch.randn(50, 21),
        pos_P=torch.randn(50, 3),
        pocket_center=torch.zeros(3)
    )
    batch = Batch.from_data_list([data]).to(device)
    batch.pos_L.requires_grad_(True)
    
    # 3. Forward Pass with t=0.5
    t = torch.tensor([0.5], device=device)
    v_pred, confidence = rf.backbone(t, batch)
    
    print(f"Confidence Score: {confidence.item():.4f}")
    
    # 4. Backward to pos_L
    grad_x = torch.autograd.grad(confidence.sum(), batch.pos_L)[0]
    
    print(f"Gradient Norm: {grad_x.norm().item():.6f}")
    
    assert grad_x is not None, "Gradient should not be None"
    assert grad_x.norm() > 0, "Gradient should be non-zero for guidance to work"
    
    print("✅ SUCCESS: Confidence-Guided gradients are propagating correctly.")

if __name__ == "__main__":
    test_guidance_gradients()
