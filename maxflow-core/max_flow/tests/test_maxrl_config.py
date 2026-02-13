import unittest
import torch
import os
import json
import tempfile
from max_flow.config import MaxRLConfig
from max_flow.utils.training import DynamicRewardScaler
from max_flow.models.max_rl import MaxRL
# Mock models
class MockFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

class TestMaxRLConfig(unittest.TestCase):
    def test_save_load(self):
        config = MaxRLConfig(lr=0.01, beta=0.5)
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            config.save(tmp.name)
            tmp_path = tmp.name
            
        loaded_config = MaxRLConfig.load(tmp_path)
        self.assertEqual(loaded_config.lr, 0.01)
        self.assertEqual(loaded_config.beta, 0.5)
        self.assertEqual(loaded_config.grad_clip_norm, 1.0) # Default
        
        os.remove(tmp_path)

class TestDynamicRewardScaler(unittest.TestCase):
    def test_update_normalize(self):
        scaler = DynamicRewardScaler()
        
        # Batch 1: [10, 20, 30] -> Mean 20, Std ~10 (population)
        r1 = torch.tensor([10.0, 20.0, 30.0])
        scaler.update(r1)
        
        self.assertAlmostEqual(scaler.mean, 20.0, places=4)
        
        # Batch 2: [10, 20, 30] again
        scaler.update(r1)
        self.assertAlmostEqual(scaler.mean, 20.0, places=4)
        
        # Normalize [20.0] -> Should be 0.0
        norm = scaler.normalize(torch.tensor([20.0]))
        self.assertAlmostEqual(norm.item(), 0.0, places=4)
        
        # Normalize [30.0] -> (30-20)/std. Var should be ~66.6 or 100 depending on N-1?
        # Welford usually does population variance or sample? 
        # Code: M2 / count. This is population variance.
        # Var = ((10-20)^2 + (0)^2 + (10^2)) * 2 / 6 = 200*2/6 = 66.66
        # Std = 8.16
        # Z = 10 / 8.16 = 1.225
        
        norm_30 = scaler.normalize(torch.tensor([30.0]))
        print(f"Mean: {scaler.mean}, Std: {scaler.std}, Norm(30): {norm_30.item()}")
        self.assertTrue(abs(norm_30.item()) > 0.1)

class TestMaxRLInit(unittest.TestCase):
    def test_init_with_config(self):
        config = MaxRLConfig(beta=0.9, clip_val=5.0)
        policy = MockFlow()
        ref = MockFlow()
        MaxRL = MaxRL(policy, ref, config=config)
        
        self.assertEqual(MaxRL.beta, 0.9)
        self.assertEqual(MaxRL.clip_val, 5.0)
        self.assertEqual(MaxRL.lambda_geom, 0.01) # Default

    def test_loss_execution(self):
        # Setup minimal mock components
        config = MaxRLConfig(use_dynamic_scaling=True)
        
        # Mock Backbone output: ((v, rot), p_water, admet, charge, rmsf, chiral) 
        # or just correct tuple structure
        class MockBackbone(torch.nn.Module):
            def forward(self, t, data):
                batch_size = t.size(0)
                # Output must match unpacking in MaxRL.py/flow_matching.py
                # v_info: (v_model, p_water, admet_pred, charge_delta, rmsf_L, chiral_pred)
                v_model = torch.zeros_like(data.pos_L)
                return (v_model, None, None, None, None, None), torch.zeros(batch_size), torch.zeros(batch_size)

        class MockRF(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = MockBackbone()

        policy = MockRF()
        ref = MockRF()
        MaxRL = MaxRL(policy, ref, config=config)
        
        # Mock Data
        from torch_geometric.data import Data, Batch
        data = Data(pos_L=torch.randn(10, 3), pocket_center=torch.zeros(3), x_L_batch=torch.zeros(10, dtype=torch.long))
        batch = Batch.from_data_list([data, data]) # Batch size 2
        
        # Run Loss
        # Rewards should be normalized if passed, but here we test the flow
        # If we pass scaler, it should work
        loss = MaxRL.loss(batch, batch, reward_win=torch.tensor([1.0, 1.0]), reward_lose=torch.tensor([0.0, 0.0]))
        
        print(f"Loss Output: {loss.item()}")
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss))

if __name__ == '__main__':
    unittest.main()

if __name__ == '__main__':
    unittest.main()
