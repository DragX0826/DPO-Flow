"""
Test script for integrated optimizations.
Tests: Adaptive Loss Weighting, Smooth PSA Constraint, PCGrad Gradient Surgery
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.adaptive_loss import AdaptiveLossWeighting, MultiTaskLossWrapper
from models.smooth_psa import SmoothPSARegularizer, PSAConstraintWrapper, PSAAnalyzer
from models.pcgrad import PCGrad, MultiObjectivePCGrad
from training.integrated_trainer import IntegratedReflowTrainer, IntegratedTrainingConfig


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        if isinstance(x, dict):
            x = x.get('input', torch.randn(1, 10))
        
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return {
            'output': x,
            'psa': torch.randn(x.shape[0]) * 15 + 75,  # Simulated PSA values
            'uncertainty': torch.randn(x.shape[0]) * 0.1 + 0.5  # Simulated uncertainty
        }


def test_adaptive_loss():
    """Test adaptive loss weighting."""
    print("🧪 Testing Adaptive Loss Weighting...")
    
    # Create adaptive loss
    adaptive_loss = AdaptiveLossWeighting(num_losses=4)
    
    # Create dummy losses
    losses = [
        torch.tensor(1.0, requires_grad=True),
        torch.tensor(2.0, requires_grad=True),
        torch.tensor(0.5, requires_grad=True),
        torch.tensor(1.5, requires_grad=True)
    ]
    
    # Calculate weighted loss
    total_loss, metrics = adaptive_loss.calculate_total_loss(losses)
    
    print(f"✅ Total loss: {total_loss.item():.4f}")
    print(f"✅ Adaptive weights: {metrics['weights']}")
    print(f"✅ Uncertainty regularization: {metrics['uncertainty_regularization']:.4f}")
    
    return True


def test_smooth_psa():
    """Test smooth PSA constraint."""
    print("\n🧪 Testing Smooth PSA Constraint...")
    
    # Create PSA regularizer
    psa_regularizer = SmoothPSARegularizer(target_psa=75.0, sigma=15.0, weight=1.0)
    
    # Test with different PSA values
    test_psa_values = torch.tensor([50.0, 75.0, 100.0, 150.0])
    
    loss, metrics = psa_regularizer(test_psa_values)
    
    print(f"✅ PSA loss: {loss.item():.4f}")
    print(f"✅ Mean PSA: {metrics['mean_psa']:.2f}")
    print(f"✅ Percentage within range: {metrics['percentage_within_range']:.2f}%")
    
    # Test gradient
    gradient = psa_regularizer.get_gradient(test_psa_values)
    print(f"✅ Gradient shape: {gradient.shape}")
    print(f"✅ Mean gradient: {torch.mean(gradient).item():.4f}")
    
    # Test analyzer
    analyzer = PSAAnalyzer(target_psa=75.0, sigma=15.0)
    analysis = analyzer.analyze_distribution(test_psa_values)
    
    print(f"✅ Analysis keys: {list(analysis.keys())}")
    
    return True


def test_pcgrad():
    """Test PCGrad gradient surgery."""
    print("\n🧪 Testing PCGrad Gradient Surgery...")
    
    # Create PCGrad
    pc_grad = PCGrad()
    
    # Create conflicting gradients
    grad1 = torch.tensor([1.0, 0.5, -0.3])
    grad2 = torch.tensor([-0.8, 1.2, 0.6])
    grad3 = torch.tensor([0.3, -0.7, 1.1])
    
    gradients = [grad1, grad2, grad3]
    
    # Apply PCGrad
    projected_gradients = pc_grad.pc_gradient(gradients)
    
    print(f"✅ Original gradients: {[g.tolist() for g in gradients]}")
    print(f"✅ Projected gradients: {[g.tolist() for g in projected_gradients]}")
    
    # Test conflict analysis
    conflict_analysis = pc_grad.get_conflict_analysis()
    print(f"✅ Conflict analysis: {conflict_analysis}")
    
    # Test gradient analysis
    gradient_analysis = pc_grad.get_gradient_analysis()
    print(f"✅ Gradient analysis: {gradient_analysis}")
    
    return True


def test_integrated_trainer():
    """Test integrated trainer."""
    print("\n🧪 Testing Integrated Trainer...")
    
    # Create model
    model = SimpleModel()
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create config
    config = IntegratedTrainingConfig.get_default_config()
    
    # Create integrated trainer
    trainer = IntegratedReflowTrainer(model, optimizer, config)
    
    # Create dummy batch
    batch = {
        'input': torch.randn(32, 10),
        'target': torch.randn(32, 1)
    }
    
    # Test training step
    metrics = trainer.train_step(batch)
    
    print(f"✅ Training metrics keys: {list(metrics.keys())}")
    print(f"✅ Total loss: {metrics['total_loss']:.4f}")
    print(f"✅ PSA metrics: {[(k, v) for k, v in metrics.items() if 'psa' in k]}")
    
    # Test validation step
    val_metrics = trainer.validate_step(batch)
    print(f"✅ Validation metrics keys: {list(val_metrics.keys())}")
    
    # Test training summary
    summary = trainer.get_training_summary()
    print(f"✅ Training summary keys: {list(summary.keys())}")
    
    return True


def test_integration():
    """Test full integration."""
    print("\n🧪 Testing Full Integration...")
    
    # Create model
    model = SimpleModel()
    
    # Create data loaders
    class DummyDataLoader:
        def __init__(self, batch_size=32, num_batches=10):
            self.batch_size = batch_size
            self.num_batches = num_batches
            
        def __iter__(self):
            for i in range(self.num_batches):
                yield {
                    'input': torch.randn(self.batch_size, 10),
                    'target': torch.randn(self.batch_size, 1)
                }
        
        def __len__(self):
            return self.num_batches
    
    train_loader = DummyDataLoader()
    val_loader = DummyDataLoader()
    
    # Create config
    config = {
        'num_epochs': 2,
        'learning_rate': 1e-3,
        'log_every': 5,
        'save_every': 10,
        **IntegratedTrainingConfig.get_default_config()
    }
    
    # Import training function
    from training.integrated_trainer import train_with_integrated_optimizations
    
    # Train model
    trainer = train_with_integrated_optimizations(
        model, train_loader, val_loader, config
    )
    
    print("✅ Full integration test completed")
    
    return True


def run_performance_benchmark():
    """Run performance benchmark."""
    print("\n🚀 Running Performance Benchmark...")
    
    # Create model
    model = SimpleModel()
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create config
    config = IntegratedTrainingConfig.get_default_config()
    
    # Create integrated trainer
    trainer = IntegratedReflowTrainer(model, optimizer, config)
    
    # Create dummy batch
    batch = {
        'input': torch.randn(64, 10),
        'target': torch.randn(64, 1)
    }
    
    # Warm up
    for _ in range(10):
        trainer.train_step(batch)
    
    # Benchmark
    import time
    num_steps = 100
    
    start_time = time.time()
    for _ in range(num_steps):
        trainer.train_step(batch)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_steps
    
    print(f"✅ Average training step time: {avg_time:.4f} seconds")
    print(f"✅ Steps per second: {1.0 / avg_time:.2f}")
    
    return True


def main():
    """Main test function."""
    print("🔬 Starting Integrated Optimizations Test Suite")
    print("=" * 60)
    
    tests = [
        ("Adaptive Loss Weighting", test_adaptive_loss),
        ("Smooth PSA Constraint", test_smooth_psa),
        ("PCGrad Gradient Surgery", test_pcgrad),
        ("Integrated Trainer", test_integrated_trainer),
        ("Full Integration", test_integration),
        ("Performance Benchmark", run_performance_benchmark)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*60}")
            print(f"Running: {test_name}")
            print(f"{'='*60}")
            
            success = test_func()
            
            if success:
                print(f"✅ {test_name} - PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} - FAILED")
                failed += 1
                
        except Exception as e:
            print(f"❌ {test_name} - FAILED with exception: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Test Summary:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {passed/(passed+failed)*100:.1f}%")
    print(f"{'='*60}")
    
    if failed == 0:
        print("🎉 All tests passed! Integrated optimizations are working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
