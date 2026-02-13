import torch
import numpy as np
import matplotlib.pyplot as plt

def fractional_diff(x, d, window=10):
    """
    Grunwald-Letnikov Fractional Differentiation.
    x: (T, D) sequence of features.
    d: differentiation order (e.g. 0.5)
    """
    weights = [1.0]
    for k in range(1, window):
        weights.append(weights[-1] * (k - d - 1) / k)
    
    weights = torch.tensor(weights, device=x.device).flip(0)
    
    res = []
    for i in range(window, len(x)):
        chunk = x[i-window:i]
        diff_val = (chunk * weights.unsqueeze(-1)).sum(dim=0)
        res.append(diff_val)
    
    return torch.stack(res)

def test_fractional_utility():
    print("Testing Fractional Differentiation Utility (Quant-Inspired)...")
    
    # 1. Simulate a Non-Stationary Geometric Signal (e.g. Atom moving towards pocket)
    t = torch.linspace(0, 10, 100)
    signal = 2.0 * t + torch.randn(100) * 0.5 # Linear trend + noise
    signal = signal.view(-1, 1)
    
    # 2. Apply Differentiation
    d05 = fractional_diff(signal, d=0.5)
    d10 = signal[1:] - signal[:-1] # Integer diff (Order 1)
    
    print(f"Original Signal Std: {signal.std().item():.4f} (Non-stationary)")
    print(f"Order 1.0 Diff Std: {d10.std().item():.4f} (Over-differentiated, lost memory)")
    print(f"Order 0.5 Diff Std: {d05.std().item():.4f} (Fractional: Stationary yet preserves trend-memory)")
    
    # Check "Memory" - Correlation with original trend
    corr_05 = np.corrcoef(signal[10:].flatten(), d05.flatten())[0, 1]
    corr_10 = np.corrcoef(signal[1:].flatten(), d10.flatten())[0, 1]
    
    print(f"Correlation with original trend (Order 0.5): {corr_05:.4f}")
    print(f"Correlation with original trend (Order 1.0): {corr_10:.4f}")
    
    if corr_05 > corr_10:
        print("✅ SUCCESS: Fractional Differentiation (d=0.5) successfully keeps more memory than integer diff.")
    else:
        print("❌ FAIL: Fractional diff did not show memory advantage in this setup.")

if __name__ == "__main__":
    test_fractional_utility()
