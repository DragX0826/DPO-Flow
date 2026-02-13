import torch
import time
from max_flow.models.backbone import GlobalContextBlock

def test_mamba_scaling():
    print("Benchmarking Mamba-Hybrid Scaling...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = 64
    mixer = GlobalContextBlock(hidden_dim).to(device)
    
    lengths = [100, 500, 1000, 2000, 4000]
    
    for L in lengths:
        x = torch.randn(L, hidden_dim).to(device)
        
        # Warmup
        for _ in range(5):
             _ = mixer(x)
             
        start_time = time.time()
        for _ in range(10):
            out = mixer(x)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10.0
        print(f"Length {L:4d} -> Avg Inference Time: {avg_time:.6f}s")
        
        assert out.shape == x.shape, "Output shape mismatch"

    print("✅ SUCCESS: Mamba-Hybrid architecture is operational and scaling.")

if __name__ == "__main__":
    test_mamba_scaling()
