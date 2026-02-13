# max_flow/tests/test_infra.py

import sys
import os
import torch
from torch_geometric.data import Data, Batch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from max_flow.models.backbone import CrossGVP
from max_flow.models.flow_matching import RectifiedFlow

def test_backbone_forward():
    print("Testing Backbone Forward...")
    # Mock Data
    num_L = 20
    num_P = 100
    dim_n = 45 # Feature dim from utils/chem
    dim_h = 64
    
    data = Data(x_L=torch.randn(num_L, dim_n),
                pos_L=torch.randn(num_L, 3),
                x_P=torch.randn(num_P, 21),
                pos_P=torch.randn(num_P, 3),
                pocket_center=torch.randn(1, 3))
    
    # Instantiate Model
    backbone = CrossGVP(node_in_dim=dim_n, hidden_dim=dim_h, num_layers=2)
    
    # Forward
    t = torch.tensor([0.5]) # Single time
    v = backbone(t, data)
    
    assert v.shape == (num_L, 3)
    print("Backbone Forward Passed!")

def test_flow_loss():
    print("Testing Flow Loss...")
    num_L = 20
    num_P = 100
    dim_n = 161
    dim_h = 64
    
    data = Data(x_L=torch.randn(num_L, dim_n),
                pos_L=torch.randn(num_L, 3), # x_1
                x_P=torch.randn(num_P, 21),
                pos_P=torch.randn(num_P, 3),
                pocket_center=torch.randn(1, 3)) # For x_0 sampling
    
    # Create Batch (size 2)
    data_list = [data, data]
    batch = Batch.from_data_list(data_list)
    
    # Manual batch injection because PyG might not infer x_L/x_P as node features
    batch.x_L_batch = torch.cat([torch.zeros(num_L, dtype=torch.long), 
                                 torch.ones(num_L, dtype=torch.long)])
    batch.x_P_batch = torch.cat([torch.zeros(num_P, dtype=torch.long), 
                                 torch.ones(num_P, dtype=torch.long)])
    
    # Instantiate Model
    backbone = CrossGVP(node_in_dim=dim_n, hidden_dim=dim_h, num_layers=2)
    flow = RectifiedFlow(backbone)
    
    # Loss
    loss = flow.loss(batch)
    print(f"Loss: {loss.item()}")
    assert not torch.isnan(loss)
    print("Flow Loss Passed!")

def test_sampling():
    print("Testing Sampling...")
    num_L = 20
    num_P = 50
    dim_n = 45
    dim_h = 32
    
    data = Data(x_L=torch.randn(num_L, dim_n),
                pos_L=torch.randn(num_L, 3), # Dummy, overwritten during sample
                x_P=torch.randn(num_P, 21),
                pos_P=torch.randn(num_P, 3),
                pocket_center=torch.randn(1, 3))
    
    backbone = CrossGVP(node_in_dim=dim_n, hidden_dim=dim_h, num_layers=1)
    flow = RectifiedFlow(backbone)
    
    x_final, traj = flow.sample(data, steps=5)
    
    assert x_final.shape == (num_L, 3)
    assert len(traj) == 6 # Init + 5 steps
    print("Sampling Passed!")

if __name__ == "__main__":
    import traceback
    with open("test_debug_output.txt", "w") as f:
        try:
            test_backbone_forward()
            test_flow_loss()
            test_sampling()
            f.write("ALL TESTS PASSED\n")
        except Exception:
            f.write("ERROR DETECTED\n")
            traceback.print_exc(file=f)
    
    # Also print to stdout for convenience
    if os.path.exists("test_debug_output.txt"):
        with open("test_debug_output.txt", "r") as f:
            print(f.read())
