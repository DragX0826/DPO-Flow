#!/usr/bin/env python3
"""
Full Integration Test for Enhanced Optimizations

This script tests the complete integration of:
1. Homoscedastic Uncertainty Weighting (Adaptive Loss)
2. Soft Gaussian Potential (Smooth PSA)
3. PCGrad (Gradient Surgery)
4. 1-Step Rectified Flow
5. Uncertainty-Aware Reward

Usage:
    python test_full_integration.py --test_mode --num_epochs 1 --batch_size 2
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from max_flow.models.adaptive_loss import AdaptiveLossWeighting, MultiTaskLossWrapper
from max_flow.models.smooth_psa import SmoothPSARegularizer
from max_flow.models.pcgrad import PCGrad, MultiObjectivePCGrad
from max_flow.training.integrated_trainer import IntegratedReflowTrainer
def calculate_quality_score(batch):
    """Calculate quality score for batch"""
    # Simple quality calculation based on molecular properties
    qed_score = torch.sigmoid((batch['qed'] - 0.5) / 0.2)  # Normalize QED
    sa_score = torch.sigmoid((4.0 - batch['sa']) / 1.0)  # Lower SA is better
    tpsa_score = torch.exp(-((batch['tpsa'] - 75.0) ** 2) / (2 * 15 ** 2))  # Gaussian around 75
    
    # Combined score
    quality_score = 0.4 * qed_score + 0.3 * sa_score + 0.3 * tpsa_score
    
    return quality_score.mean()


class MockReflowModel(nn.Module):
    """Mock model for testing integration"""
    
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, batch, timesteps=None, **kwargs):
        """Forward pass - handle batch dictionary"""
        # Extract input from batch
        if isinstance(batch, dict):
            x = batch.get('x_0', batch.get('x_L', torch.randn(1, self.input_dim)))
            if x.dim() == 1:
                x = x.unsqueeze(0)
        else:
            x = batch
            if x.dim() == 1:
                x = x.unsqueeze(0)
        
        # Return dictionary with predictions
        output = self.encoder(x)
        return {
            'pred': output,
            'x_pred': output,
            'psa': torch.tensor([[75.0]], requires_grad=True),  # Mock PSA prediction
            'quality': torch.tensor([[0.8]], requires_grad=True)  # Mock quality prediction
        }


class MockDataset(torch.utils.data.Dataset):
    """Mock dataset for testing"""
    
    def __init__(self, num_samples=100, input_dim=128):
        self.num_samples = num_samples
        self.input_dim = input_dim
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        """Generate mock data"""
        x_0 = torch.randn(self.input_dim)  # Noise
        x_1 = torch.randn(self.input_dim)  # Data
        tpsa = torch.tensor(65.0 + 20 * torch.randn(1).item())  # PSA around 65±20
        qed = torch.tensor(0.7 + 0.2 * torch.randn(1).item())  # QED around 0.7±0.2
        sa = torch.tensor(3.0 + 1.0 * torch.randn(1).item())  # SA around 3±1
        
        return {
            'x_0': x_0,
            'x_1': x_1,
            'tpsa': tpsa,
            'qed': qed,
            'sa': sa,
            'batch_size': 1
        }


def test_adaptive_loss_integration():
    """Test adaptive loss weighting integration"""
    print("🧪 Testing Adaptive Loss Weighting...")
    
    # Create adaptive loss weighting
    num_losses = 3
    adaptive_loss = AdaptiveLossWeighting(num_losses, initial_log_sigma=-1.0)
    
    # Create mock losses
    losses = [
        torch.tensor(0.5, requires_grad=True),
        torch.tensor(0.3, requires_grad=True),
        torch.tensor(0.7, requires_grad=True)
    ]
    
    # Calculate total loss
    total_loss, metrics = adaptive_loss.calculate_total_loss(losses)
    
    # Verify gradients flow
    total_loss.backward()
    
    # Check that log_sigma has gradients
    assert adaptive_loss.log_sigma.grad is not None, "log_sigma should have gradients"
    assert adaptive_loss.log_sigma.grad.abs().sum() > 0, "log_sigma gradients should be non-zero"
    
    print(f"✅ Adaptive Loss: total_loss={total_loss.item():.4f}")
    print(f"   Weights: {metrics['weights']}")
    print(f"   Log Sigma: {metrics['log_sigma']}")
    
    return True


def test_smooth_psa_integration():
    """Test smooth PSA regularization integration"""
    print("🧪 Testing Smooth PSA Regularization...")
    
    # Create smooth PSA regularizer
    psa_regularizer = SmoothPSARegularizer(
        target_psa=75.0,
        sigma=15.0,
        weight=0.1
    )
    
    # Test different PSA values
    test_psa_values = [50.0, 75.0, 100.0, 65.0, 85.0]
    
    for psa in test_psa_values:
        loss, metrics = psa_regularizer(torch.tensor(psa))
        print(f"   PSA={psa:5.1f} -> Loss={loss.item():.6f}")
    
    # Verify it's smooth and differentiable
    psa_tensor = torch.tensor(75.0, requires_grad=True)
    loss, metrics = psa_regularizer(psa_tensor)
    loss.backward()
    
    assert psa_tensor.grad is not None, "PSA tensor should have gradients"
    print(f"✅ Smooth PSA: Gradients work correctly")
    
    return True


def test_pcgrad_integration():
    """Test PCGrad integration"""
    print("🧪 Testing PCGrad Integration...")
    
    # Create simple model
    model = nn.Linear(10, 5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create PCGrad 
    pcgrad = PCGrad(reduction='mean')
    
    # Create conflicting gradients
    x = torch.randn(4, 10)
    y1 = model(x)  # Task 1
    y2 = -model(x)  # Task 2 (conflicting)
    
    loss1 = y1.sum()
    loss2 = y2.sum()
    
    # Calculate gradients
    grad1 = torch.autograd.grad(loss1, model.parameters(), retain_graph=True, create_graph=True)
    grad2 = torch.autograd.grad(loss2, model.parameters(), retain_graph=True, create_graph=True)
    
    # Apply PCGrad
    gradients = [torch.cat([g.flatten() for g in grad1]), torch.cat([g.flatten() for g in grad2])]
    projected_gradients = pcgrad.pc_gradient(gradients)
    
    print(f"✅ PCGrad: Applied to conflicting gradients")
    
    return True


def test_integrated_trainer():
    """Test full integrated trainer"""
    print("🧪 Testing Integrated Trainer...")
    
    # Create mock model and dataset
    model = MockReflowModel(input_dim=128)
    dataset = MockDataset(num_samples=20, input_dim=128)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Create trainer config
    config = {
        'num_tasks': 3,
        'learning_rate': 1e-4,
        'use_pcgrad': True,
        'use_adaptive_loss': True,
        'use_smooth_psa': True,
        'device': 'cpu'
    }
    
    # Create trainer
    trainer = IntegratedReflowTrainer(
        model=model,
        optimizer=optim.Adam(model.parameters(), lr=config['learning_rate']),
        config=config
    )
    
    # Run one training step
    for batch in dataloader:
        metrics = trainer.train_step(batch)
        break
    
    # Verify metrics
    required_keys = ['total_loss', 'consistency', 'psa', 'quality', 'uncertainty', 'weights']
    for key in required_keys:
        assert key in metrics, f"Missing metric: {key}"
    
    print(f"✅ Integrated Trainer: All metrics present")
    print(f"   Total Loss: {metrics['total_loss']:.6f}")
    print(f"   Consistency Loss: {metrics['consistency']:.6f}")
    print(f"   PSA Loss: {metrics['psa']:.6f}")
    print(f"   Quality Loss: {metrics['quality']:.6f}")
    print(f"   Uncertainty Loss: {metrics['uncertainty']:.6f}")
    print(f"   Adaptive Weights: {metrics['weights'][:3]}")
    
    return True


def test_checkpoint_saving():
    """Test checkpoint saving with all components"""
    print("🧪 Testing Checkpoint Saving...")
    
    # Create trainer config
    config = {
        'num_tasks': 3,
        'use_adaptive_loss': True,
        'device': 'cpu'
    }
    
    # Create trainer
    model = MockReflowModel()
    trainer = IntegratedReflowTrainer(
        model=model,
        optimizer=optim.Adam(model.parameters()),
        config=config
    )
    
    # Create mock metrics
    metrics = {
        'total_loss': 1.5,
        'flow_loss': 0.5,
        'psa_loss': 0.3,
        'quality_loss': 0.7,
        'adaptive_weights': [0.4, 0.3, 0.3]
    }
    
    # Test checkpoint saving
    checkpoint_path = "/tmp/test_checkpoint.pt"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    trainer.save_checkpoint(checkpoint_path, metrics)
    
    # Verify checkpoint exists and can be loaded
    assert os.path.exists(checkpoint_path), "Checkpoint file should exist"
    
    checkpoint = torch.load(checkpoint_path)
    required_keys = ['model_state_dict', 'optimizer_state_dict', 'adaptive_loss_state_dict', 'metrics']
    for key in required_keys:
        assert key in checkpoint, f"Missing checkpoint key: {key}"
    
    print(f"✅ Checkpoint Saving: All components saved correctly")
    
    # Cleanup
    os.remove(checkpoint_path)
    
    return True


def test_quality_assessment():
    """Test quality assessment integration"""
    print("🧪 Testing Quality Assessment...")
    
    # Test with mock molecule data
    batch = {
        'tpsa': torch.tensor([65.0, 80.0, 45.0]),
        'qed': torch.tensor([0.7, 0.8, 0.5]),
        'sa': torch.tensor([3.0, 2.5, 4.0]),
    }
    
    # Calculate quality score
    quality_score = calculate_quality_score(batch)
    
    print(f"✅ Quality Assessment: score={quality_score.item():.4f}")
    
    return True


def run_full_integration_test(args):
    """Run complete integration test"""
    print("🚀 Running Full Integration Test...")
    print("=" * 60)
    
    tests = [
        ("Adaptive Loss", test_adaptive_loss_integration),
        ("Smooth PSA", test_smooth_psa_integration),
        ("PCGrad", test_pcgrad_integration),
        ("Quality Assessment", test_quality_assessment),
        ("Integrated Trainer", test_integrated_trainer),
        ("Checkpoint Saving", test_checkpoint_saving),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success, None))
            print(f"✅ {test_name}: PASSED")
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"❌ {test_name}: FAILED - {e}")
        print("-" * 40)
    
    # Summary
    print("\n📊 Integration Test Summary:")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, error in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if error and args.verbose:
            print(f"   Error: {error}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        print("✅ Task 12: Integration of all optimizations - COMPLETED")
        return True
    else:
        print("⚠️  Some tests failed. Review errors above.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test full integration of enhanced optimizations")
    parser.add_argument("--test_mode", action="store_true", help="Run in test mode")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    success = run_full_integration_test(args)
    
    if success:
        print("\n🎯 Next Steps:")
        print("1. Update documentation (Task 13)")
        print("2. Run performance benchmarks")
        print("3. Deploy to production pipeline")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
