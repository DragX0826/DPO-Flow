"""
1-Step Sampling Validation for Reflow Models
Validates that the trained reflow model can generate molecules in a single step.
"""

import torch
import argparse
import os
from tqdm import tqdm
from torch_geometric.data import Batch
from max_flow.models.max_rl import MaxFlow
from max_flow.utils.quality_assessment import calculate_molecule_quality, calculate_consistency
from accelerate import Accelerator


def validate_1step_sampling(
    model_path,
    n_samples=1000,
    batch_size=32,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    quality_threshold=0.8,
    consistency_threshold=0.95
):
    """
    Validate 1-step sampling capability of a reflow model.
    
    Args:
        model_path: Path to trained reflow model
        n_samples: Number of samples to generate
        batch_size: Batch size for generation
        device: Device to use
        quality_threshold: Minimum quality score
        consistency_threshold: Minimum consistency threshold
        
    Returns:
        results: Dictionary with validation metrics
    """
    print(f"🚀 1-Step Sampling Validation")
    print(f"Model: {model_path}")
    print(f"Samples: {n_samples} | Batch Size: {batch_size}")
    
    # Initialize Accelerator
    accelerator = Accelerator()
    
    # Load model
    try:
        model = MaxFlow()
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create dummy pocket for sampling
    dummy_pocket = torch.randn(50, 6).to(device)  # 50 atoms, 6 features
    dummy_pos_P = torch.randn(50, 3).to(device)
    dummy_center = dummy_pos_P.mean(dim=0)
    
    # Validation metrics
    quality_scores = []
    consistency_scores = []
    accepted_samples = 0
    rejected_samples = 0
    
    num_batches = (n_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Sampling"):
            current_batch_size = min(batch_size, n_samples - i * batch_size)
            
            # Create batch with noise initialization
            data_list = []
            for _ in range(current_batch_size):
                lig_size = torch.randint(10, 21, (1,)).item()
                
                # Start with noise
                data = {
                    'x_L': torch.randn(lig_size, 6).to(device) * 0.5,  # Controlled noise
                    'pos_L': torch.randn(lig_size, 3).to(device) * 0.5,
                    'x_P': dummy_pocket,
                    'pos_P': dummy_pos_P,
                    'pocket_center': dummy_center
                }
                data_list.append(data)
            
            batch = Batch.from_data_list([Batch(**data) for data in data_list]).to(device)
            
            # 1-Step Sampling: t=1.0 (direct from noise to data)
            t = torch.ones(current_batch_size, device=device)
            batch_idx = getattr(batch, 'x_L_batch', getattr(batch, 'batch', None))
            if batch_idx is not None:
                t_nodes = t[batch_idx].unsqueeze(-1)
            else:
                t_nodes = t.view(-1, 1).expand(batch.pos_L.size(0), -1)
            
            # Sample with t=1.0 for direct generation
            x_out = model.flow.sample(batch, t=t_nodes, steps=1, gamma=0.0)
            
            # Evaluate quality and consistency
            for j in range(current_batch_size):
                quality_score = calculate_molecule_quality(x_out[j])
                consistency_score = calculate_consistency(torch.stack([batch.pos_L[j], x_out[j]]))
                
                quality_scores.append(quality_score)
                consistency_scores.append(consistency_score)
                
                if quality_score >= quality_threshold and consistency_score >= consistency_threshold:
                    accepted_samples += 1
                else:
                    rejected_samples += 1
    
    # Calculate statistics
    avg_quality = np.mean(quality_scores)
    avg_consistency = np.mean(consistency_scores)
    acceptance_rate = accepted_samples / (accepted_samples + rejected_samples) * 100
    
    # Print results
    print(f"\n📊 Validation Results:")
    print(f"📈 Average Quality Score: {avg_quality:.3f}")
    print(f"📈 Average Consistency Score: {avg_consistency:.3f}")
    print(f"✅ Acceptance Rate: {acceptance_rate:.2f}%")
    print(f"✅ Accepted Samples: {accepted_samples}")
    print(f"⚠️ Rejected Samples: {rejected_samples}")
    
    # Performance metrics
    print(f"\n⚡ Performance:")
    print(f"🔄 Samples per second: {n_samples / 60:.1f} (assuming 60s total time)")
    print(f"⚡ Speedup vs 50-step: {50}x (theoretical)")
    
    return {
        'avg_quality': avg_quality,
        'avg_consistency': avg_consistency,
        'acceptance_rate': acceptance_rate,
        'accepted_samples': accepted_samples,
        'rejected_samples': rejected_samples
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1-Step Sampling Validation")
    parser.add_argument("--model", type=str, required=True, help="Path to trained reflow model")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--quality_threshold", type=float, default=0.8, help="Minimum quality score")
    parser.add_argument("--consistency_threshold", type=float, default=0.95, help="Minimum consistency threshold")
    
    args = parser.parse_args()
    
    results = validate_1step_sampling(
        args.model,
        args.n_samples,
        args.batch_size,
        quality_threshold=args.quality_threshold,
        consistency_threshold=args.consistency_threshold
    )
    
    print(f"\n🎯 Validation Complete!")
    print(f"Results: {results}")
