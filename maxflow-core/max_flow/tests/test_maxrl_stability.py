# max_flow/tests/test_MaxRL_stability.py

import torch
import torch.nn.functional as F
from max_flow.models.max_rl import MaxRL, MaxFlow
from max_flow.config import MaxRLConfig
from torch_geometric.data import Data, Batch

def test_sMaxRL_logit_clipping():
    print("🧪 Testing SMaxRL Logit Clipping...")
    model = MaxFlow()
    ref = MaxFlow()
    ref.load_state_dict(model.state_dict())
    
    # Initialize with small clip_val
    # Initialize with config object as per Phase 63 signature
    config = MaxRLConfig(beta=100.0, clip_val=2.0)
    MaxRL = MaxRL(model.flow, ref.flow, config=config)
    
    # Create mock data
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
    
    # Extreme rewards to force high logits
    reward_win = torch.tensor([1e6])
    reward_lose = torch.tensor([-1e6])
    
    # Run loss - Phase 63 returns (loss, loss_dict)
    loss, loss_dict = MaxRL.loss(batch, batch, reward_win=reward_win, reward_lose=reward_lose)
    print(f"  Loss with Extreme Rewards (Clipped): {loss.item():.4f}")
    
    # Check gradients
    loss.backward()
    max_grad = 0.0
    for p in model.parameters():
        if p.grad is not None:
            max_grad = max(max_grad, p.grad.abs().max().item())
    
    print(f"  Max Gradient: {max_grad:.4f}")
    assert torch.isfinite(torch.tensor(max_grad)) # Must not be NaN or Inf

def test_manifold_anchoring():
    print("\n🧪 Testing Manifold Anchoring Loss...")
    model = MaxFlow()
    ref = MaxFlow()
    
    # lambda_anchor=10.0 to make it dominant
    config = MaxRLConfig(lambda_anchor=10.0)
    MaxRL = MaxRL(model.flow, ref.flow, config=config)
    
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
    
    # Run loss - Phase 63 returns (loss, loss_dict)
    loss, loss_dict = MaxRL.loss(batch, batch)
    
    print(f"  Anchored Loss Total: {loss.item():.4f}")
    print(f"  Loss Components: {loss_dict}")
    
    assert loss.item() > 0
    assert 'clip' in loss_dict
    assert 'anchor' in loss_dict
    assert 'MaxRL' in loss_dict

if __name__ == "__main__":
    test_sMaxRL_logit_clipping()
    test_manifold_anchoring()
