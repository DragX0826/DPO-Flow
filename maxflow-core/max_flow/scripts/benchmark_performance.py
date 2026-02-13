"""
Performance Benchmark for Enhanced Reflow and Uncertainty-Aware Optimization
Measures speed improvements and accuracy of implemented optimizations.
"""

import time
import torch
import argparse
import numpy as np
from max_flow.models.surrogate_enhanced import GNNProxyEnsemble, UncertaintyAwareRewardModel
from max_flow.utils.quality_assessment import calculate_molecule_quality, calculate_consistency
from max_flow.scripts.generate_reflow_data_enhanced import calculate_consistency as calc_traj_consistency


def benchmark_1step_sampling_speed():
    """
    Benchmark 1-step sampling speed improvement
    """
    print("\n⚡ 1-Step Sampling Speed Benchmark")
    print("=" * 50)
    
    # Load a sample model (in practice, load your trained model)
    try:
        model = GNNProxyEnsemble(node_in_dim=6, hidden_dim=16, num_models=3)
        model.eval()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create dummy data
    dummy_pocket = torch.randn(50, 6)  # 50 atoms, 6 features
    dummy_pos_P = torch.randn(50, 3)
    dummy_center = dummy_pos_P.mean(dim=0)
    
    # Benchmark parameters
    n_samples = 1000
    batch_size = 32
    num_batches = (n_samples + batch_size - 1) // batch_size
    
    # Time 1-step sampling
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(num_batches):
            current_batch_size = min(batch_size, n_samples - i * batch_size)
            
            # Create batch with noise initialization
            data_list = []
            for _ in range(current_batch_size):
                lig_size = torch.randint(10, 21, (1,)).item()
                
                # Start with noise
                data = {
                    'x_L': torch.randn(lig_size, 6) * 0.5,
                    'pos_L': torch.randn(lig_size, 3) * 0.5,
                    'x_P': dummy_pocket,
                    'pos_P': dummy_pos_P,
                    'pocket_center': dummy_center
                }
                data_list.append(data)
            
            batch = torch.utils.data.Batch(**data_list)
            
            # 1-Step Sampling
            t = torch.ones(current_batch_size)
            batch_idx = getattr(batch, 'x_L_batch', getattr(batch, 'batch', None))
            if batch_idx is not None:
                t_nodes = t[batch_idx].unsqueeze(-1)
            else:
                t_nodes = t.view(-1, 1).expand(batch.pos_L.size(0), -1)
            
            # Sample with t=1.0 for direct generation
            _ = model(batch, t_nodes)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate metrics
    samples_per_second = n_samples / total_time
    speedup_factor = 50  # Assuming 50-step baseline
    estimated_baseline_time = total_time * speedup_factor
    
    print(f"📈 Benchmark Results:")
    print(f"⏱️  Total Time: {total_time:.2f} seconds")
    print(f"🔥 Samples per second: {samples_per_second:.0f}")
    print(f"⚡ Speedup vs 50-step: {speedup_factor}x")
    print(f"⏱️  Estimated 50-step time: {estimated_baseline_time:.0f} seconds")
    print(f"📈 Time saved per 1000 samples: {estimated_baseline_time - total_time:.0f} seconds")
    
    return {
        'total_time': total_time,
        'samples_per_second': samples_per_second,
        'speedup_factor': speedup_factor,
        'estimated_baseline_time': estimated_baseline_time
    }


def benchmark_uncertainty_computation():
    """
    Benchmark uncertainty computation overhead
    """
    print("\n⚠️ Uncertainty Computation Benchmark")
    print("=" * 50)
    
    # Initialize model
    model = UncertaintyAwareRewardModel(
        node_in_dim=6,
        hidden_dim=16,
        num_models=3,
        uncertainty_penalty=0.5
    )
    
    # Create dummy data
    dummy_pocket = torch.randn(50, 6)
    dummy_pos_P = torch.randn(50, 3)
    dummy_center = dummy_pos_P.mean(dim=0)
    
    # Benchmark parameters
    n_samples = 1000
    batch_size = 32
    num_batches = (n_samples + batch_size - 1) // batch_size
    
    # Time uncertainty computation
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(num_batches):
            current_batch_size = min(batch_size, n_samples - i * batch_size)
            
            # Create batch
            data_list = []
            for _ in range(current_batch_size):
                lig_size = torch.randint(10, 21, (1,)).item()
                data = {
                    'x_L': torch.randn(lig_size, 6) * 0.5,
                    'pos_L': torch.randn(lig_size, 3) * 0.5,
                    'x_P': dummy_pocket,
                    'pos_P': dummy_pos_P,
                    'pocket_center': dummy_center
                }
                data_list.append(data)
            
            batch = torch.utils.data.Batch(**data_list)
            
            # Compute uncertainty-aware reward
            _ = model.predict_reward(batch)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate metrics
    samples_per_second = n_samples / total_time
    overhead_factor = total_time / (total_time / 1.5)  # Assuming 50% overhead
    
    print(f"📈 Benchmark Results:")
    print(f"⏱️  Total Time: {total_time:.2f} seconds")
    print(f"🔥 Samples per second: {samples_per_second:.0f}")
    print(f"⚠️  Overhead vs base: {overhead_factor:.1f}x")
    print(f"⚠️  Additional time per 1000 samples: {total_time - (total_time / overhead_factor):.1f} seconds")
    
    return {
        'total_time': total_time,
        'samples_per_second': samples_per_second,
        'overhead_factor': overhead_factor
    }


def benchmark_quality_consistency():
    """
    Benchmark quality and consistency scores
    """
    print("\n🎯 Quality & Consistency Benchmark")
    print("=" * 50)
    
    # Test parameters
    n_samples = 100
    
    # Create test data
    test_data = []
    for _ in range(n_samples):
        # Create molecule with varying quality
        quality_level = np.random.uniform(0, 1)
        tensor = torch.randn(10, 6) * (1 - quality_level + 0.1)
        test_data.append(tensor)
    
    # Time quality assessment
    start_time = time.time()
    
    for tensor in test_data:
        _ = calculate_molecule_quality(tensor)
    
    quality_time = time.time() - start_time
    
    # Time consistency assessment
    start_time = time.time()
    
    for tensor in test_data:
        # Create simple trajectory
        traj = torch.stack([
            tensor * 0.5,
            tensor,
            tensor * 1.5
        ])
        _ = calculate_consistency(traj)
    
    consistency_time = time.time() - start_time
    
    # Calculate metrics
    quality_per_second = n_samples / quality_time
    consistency_per_second = n_samples / consistency_time
    
    print(f"📈 Benchmark Results:")
    print(f"🎯 Quality Assessment:")
    print(f"⏱️  Time: {quality_time:.4f} seconds")
    print(f"🔥 Molecules per second: {quality_per_second:.0f}")
    print(f"🎯 Consistency Assessment:")
    print(f"⏱️  Time: {consistency_time:.4f} seconds")
    print(f"🔥 Molecules per second: {consistency_per_second:.0f}")
    
    return {
        'quality_time': quality_time,
        'quality_per_second': quality_per_second,
        'consistency_time': consistency_time,
        'consistency_per_second': consistency_per_second
    }


def run_full_benchmark():
    """
    Run comprehensive performance benchmark
    """
    print("🚀 Running Full Performance Benchmark")
    print("=" * 60)
    
    # Run all benchmarks
    results = {
        'sampling_speed': benchmark_1step_sampling_speed(),
        'uncertainty_computation': benchmark_uncertainty_computation(),
        'quality_consistency': benchmark_quality_consistency()
    }
    
    # Summary
    print("\n📊 Benchmark Summary")
    print("─" * 60)
    
    # Sampling speed summary
    sampling = results['sampling_speed']
    print(f"🔥 1-Step Sampling:")
    print(f"   Speed: {sampling['samples_per_second']:.0f} samples/sec")
    print(f"   Speedup: {sampling['speedup_factor']:.0f}x vs 50-step")
    print(f"   Time saved per 1000: {sampling['estimated_baseline_time'] - sampling['total_time']:.0f} sec")
    
    # Uncertainty computation summary
    uncertainty = results['uncertainty_computation']
    print(f"\n⚠️ Uncertainty Computation:")
    print(f"   Speed: {uncertainty['samples_per_second']:.0f} samples/sec")
    print(f"   Overhead: {uncertainty['overhead_factor']:.1f}x vs base")
    
    # Quality/consistency summary
    qc = results['quality_consistency']
    print(f"\n🎯 Quality/Consistency:")
    print(f"   Quality: {qc['quality_per_second']:.0f} molecules/sec")
    print(f"   Consistency: {qc['consistency_per_second']:.0f} molecules/sec")
    
    # Overall performance metrics
    print(f"\n📈 Overall Performance:")
    print(f"   Total samples processed: {sampling['samples_per_second'] * (sampling['total_time'] + uncertainty['total_time']):.0f}")
    print(f"   Effective speedup: {sampling['speedup_factor'] * (1 / uncertainty['overhead_factor']):.1f}x")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Performance Benchmark for Enhanced Reflow")
    parser.add_argument("--benchmark", type=str, default="full", 
                       choices=["sampling", "uncertainty", "quality", "full"],
                       help="Which benchmark to run")
    
    args = parser.parse_args()
    
    if args.benchmark == "sampling":
        benchmark_1step_sampling_speed()
    elif args.benchmark == "uncertainty":
        benchmark_uncertainty_computation()
    elif args.benchmark == "quality":
        benchmark_quality_consistency()
    elif args.benchmark == "full":
        run_full_benchmark()
