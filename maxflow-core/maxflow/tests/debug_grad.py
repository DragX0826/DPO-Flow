import torch
from maxflow.models.backbone import CrossGVP
from torch_geometric.data import Data, Batch

def debug_differentiability():
    print("Debugging Differentiability...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossGVP(node_in_dim=161, hidden_dim=64).to(device)
    
    num_nodes = 5
    data = Data(
        x_L=torch.randn(num_nodes, 161),
        pos_L=torch.randn(num_nodes, 3).requires_grad_(True),
        x_P=torch.randn(10, 21),
        pos_P=torch.randn(10, 3),
        pocket_center=torch.zeros(3)
    )
    batch = Batch.from_data_list([data]).to(device)
    batch.pos_L.requires_grad_(True)
    
    t = torch.tensor([0.5], device=device)
    
    # Check 1: Backbone forward
    vel, conf, kl = model(t, batch)
    
    print(f"Conf requires_grad: {conf.requires_grad}")
    print(f"Conf grad_fn: {conf.grad_fn}")
    
    if not conf.requires_grad:
        print("❌ FAIL: Confidence score is NOT differentiable!")
        # Trace back
        pass
    else:
        grad = torch.autograd.grad(conf.sum(), batch.pos_L, allow_unused=True)[0]
        if grad is None:
            print("❌ FAIL: Gradient is None (Unused)!")
        else:
            print(f"✅ SUCCESS: Gradient found! Norm: {grad.norm().item():.6f}")

if __name__ == "__main__":
    debug_differentiability()
Line:1
