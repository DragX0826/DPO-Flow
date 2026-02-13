"""
Phase 23: Hyper-Connections Benchmark Test
Tests gradient stability of HC, mHC, and DHC against baseline residual connections.
"""
import torch
import torch.nn as nn
import time
from maxflow.models.layers import HyperConnection, ManifoldConstrainedHC, DynamicHyperConnection

class DeepNetworkBaseline(nn.Module):
    """Deep network with standard residual connections."""
    def __init__(self, dim, depth):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth)])
        
    def forward(self, x):
        for layer in self.layers:
            x = x + torch.relu(layer(x))  # Standard residual
        return x

class DeepNetworkHC(nn.Module):
    """Deep network with Hyper-Connections."""
    def __init__(self, dim, depth, hc_type='hc'):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth)])
        if hc_type == 'hc':
            self.connections = nn.ModuleList([HyperConnection(dim) for _ in range(depth)])
        elif hc_type == 'mhc':
            self.connections = nn.ModuleList([ManifoldConstrainedHC(dim) for _ in range(depth)])
        else:  # dhc
            self.connections = nn.ModuleList([DynamicHyperConnection(dim) for _ in range(depth)])
        
    def forward(self, x):
        for layer, conn in zip(self.layers, self.connections):
            residual = torch.relu(layer(x))
            x = conn(x, residual)
        return x

def benchmark_gradient_stability(model, x, name):
    """Measure gradient norm stability during backprop."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    grad_norms = []
    for step in range(10):
        optimizer.zero_grad()
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        # Collect gradient norms
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)
        optimizer.step()
    
    mean_norm = sum(grad_norms) / len(grad_norms)
    std_norm = (sum((n - mean_norm)**2 for n in grad_norms) / len(grad_norms)) ** 0.5
    
    print(f"{name:20s} | Mean Grad Norm: {mean_norm:8.4f} | Std: {std_norm:8.4f} | Stability: {(1 - std_norm/mean_norm)*100:.1f}%")
    return mean_norm, std_norm

def main():
    print("=" * 70)
    print("Phase 23: Hyper-Connections Gradient Stability Benchmark")
    print("=" * 70)
    
    torch.manual_seed(42)
    dim = 64
    depth = 50  # Deep network
    batch_size = 32
    
    x = torch.randn(batch_size, dim)
    
    print(f"\nConfig: dim={dim}, depth={depth}, batch={batch_size}")
    print("-" * 70)
    
    # Baseline
    model_baseline = DeepNetworkBaseline(dim, depth)
    benchmark_gradient_stability(model_baseline, x.clone(), "Baseline Residual")
    
    # HC
    model_hc = DeepNetworkHC(dim, depth, 'hc')
    benchmark_gradient_stability(model_hc, x.clone(), "HyperConnection")
    
    # mHC
    model_mhc = DeepNetworkHC(dim, depth, 'mhc')
    benchmark_gradient_stability(model_mhc, x.clone(), "mHC (Manifold)")
    
    # DHC
    model_dhc = DeepNetworkHC(dim, depth, 'dhc')
    benchmark_gradient_stability(model_dhc, x.clone(), "DHC (Dynamic)")
    
    print("-" * 70)
    print("🧬 Higher Stability % = More stable training in deep networks")
    print("=" * 70)

if __name__ == "__main__":
    main()
