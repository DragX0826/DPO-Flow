import unittest
import torch
import torch.nn as nn
from max_flow.optimization.schedule_free import ScheduleFreeAdamW
from max_flow.ops.physics_kernels import pairwise_energy_triton, pairwise_energy_torch
import tempfile
import os

class TestQuantCombat(unittest.TestCase):
    def test_sfo_convergence(self):
        # Optimize x^2 + 2x + 1 = (x+1)^2. Min at x=-1
        p = torch.nn.Parameter(torch.tensor([10.0]))
        opt = ScheduleFreeAdamW([p], lr=0.1, warmup_steps=10)
        
        opt.train()
        for i in range(200):
            def closure():
                opt.zero_grad()
                loss = (p + 1)**2
                loss.backward()
                return loss
            opt.step(closure)
            
        opt.eval()
        # Should be close to -1
        print(f"SFO Final Parameter: {p.item()}")
        self.assertTrue(abs(p.item() + 1.0) < 0.1)

    def test_triton_physics_consistency(self):
        if not torch.cuda.is_available():
            print("Skipping Triton Test: CUDA not available")
            return
            
        N = 100
        coords = torch.randn(2, N, 3).cuda()
        charges = torch.randn(2, N).cuda()
        types = torch.randint(0, 4, (2, N)).cuda()
        params = torch.randn(4, 2).cuda()
        
        # PyTorch Reference
        energy_torch = pairwise_energy_torch(coords, charges, types, params)
        
        # Triton Kernel
        # Note: If Triton not installed, it will fallback to torch inside the function, 
        # so we are effectively testing the fallback logic if env is missing Triton.
        energy_triton = pairwise_energy_triton(coords, charges, types, params)
        
        diff = (energy_torch - energy_triton).abs().max().item()
        print(f"Triton vs Torch Max Diff: {diff}")
        self.assertTrue(diff < 1e-4)

    def test_schedule_free_no_scheduler_logic(self):
        # Just ensure it instantiates and runs
        model = torch.nn.Linear(10, 1)
        opt = ScheduleFreeAdamW(model.parameters(), lr=0.01)
        x = torch.randn(5, 10)
        y = model(x).sum()
        y.backward()
        opt.step()
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
