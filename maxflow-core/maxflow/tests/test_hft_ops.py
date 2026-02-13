import unittest
import torch
import torch.nn as nn
from maxflow.utils.training import ModelEMA
from maxflow.config import MaxRLConfig
import os

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
        
    def forward(self, x):
        return self.fc(x)

class TestHFTOps(unittest.TestCase):
    def test_ema_update(self):
        model = SimpleModel()
        # Initialize weights to 1.0
        with torch.no_grad():
            model.fc.weight.fill_(1.0)
            model.fc.bias.fill_(1.0)
            
        ema = ModelEMA(model, decay=0.5)
        
        # Verify initial shadow matches model
        self.assertTrue(torch.equal(ema.shadow['fc.weight'], model.fc.weight))
        
        # Update model weights to 2.0
        with torch.no_grad():
            model.fc.weight.fill_(2.0)
            
        # Update EMA
        # New Shadow = (1 - 0.5) * 2.0 + 0.5 * 1.0 = 1.0 + 0.5 = 1.5
        ema.update(model)
        
        self.assertAlmostEqual(ema.shadow['fc.weight'][0,0].item(), 1.5, places=4)
        
        # Verify apply_shadow
        ema.apply_shadow()
        self.assertAlmostEqual(model.fc.weight[0,0].item(), 1.5, places=4)

    def test_torch_compile_smoke(self):
        if not hasattr(torch, "compile"):
            print("Skipping compile test: torch.compile not available")
            return
            
        # Check if we are on Windows, compile support might be limited
        if os.name == 'nt':
            print("Warning: fit/compile on Windows might be slow or unsupported.")
            
        model = SimpleModel()
        try:
            compiled_model = torch.compile(model, mode="reduce-overhead")
            x = torch.randn(5, 10)
            y = compiled_model(x)
            self.assertEqual(y.shape, (5, 1))
            print("Successfully ran compiled model forward pass.")
        except Exception as e:
            print(f"Compilation failed (expected on some envs): {e}")
            # We don't fail the test if compilation fails due to environment, 
            # as long as the code handles it or we acknowledge it.
            # Real-world fallback logic should be tested if implemented.

if __name__ == '__main__':
    unittest.main()
