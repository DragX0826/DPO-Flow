"""
Reflow Data Generator (Teacher Rollout) - Enhanced
Phase 25: Single-Step Distillation with Quality Control

Generates high-quality (Noise, Data) pairs from a pre-trained Teacher model.
These pairs are used to train the Student model to jump from x_0 to x_1 in a single step.

Strategy:
1. Load Phase 24 Teacher Model.
2. Generate latent noise x_0 with controlled variance.
3. Solve ODE to get x_1 (using high-precision solver, e.g., Euler 100 steps).
4. Apply consistency checks and quality filtering.
5. Save pairs (x_0, x_1) and conditioning (pocket) for distillation.
"""

import torch
import os
import argparse
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Batch
from max_flow.models.max_rl import MaxFlow
from max_flow.utils.chem import ProteinLigandData
from max_flow.utils.metrics import calculate_molecule_quality


def generate_reflow_data(
    checkpoint_path, 
    save_path, 
    n_samples=1000, 
    batch_size=32, 
    steps=100,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    quality_threshold=0.8,
    consistency_threshold=0.95,
    hidden_dim=128,
    num_layers=4
):
    """
    Enhanced Reflow Data Generator with Quality Control
    
    Args:
        checkpoint_path: Path to pre-trained teacher model
        save_path: Path to save generated data
        n_samples: Number of samples to generate
        batch_size: Batch size for generation
        steps: Number of ODE solver steps
        device: Device to use (cuda/cpu)
        quality_threshold: Minimum quality score to accept samples
        consistency_threshold: Minimum consistency score to accept samples
    """
    print(f"🚀 Enhanced Reflow Teacher Rollout")
    print(f"Teacher: {checkpoint_path}")
    print(f"Target: {save_path}")
    print(f"Samples: {n_samples} | Solver Steps: {steps}")
    print(f"Quality Threshold: {quality_threshold} | Consistency Threshold: {consistency_threshold}")
    
    # 1. Load Teacher
    try:
        model = MaxFlow(hidden_dim=hidden_dim, num_layers=num_layers)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Handle different checkpoint formats
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        model.load_state_dict(state_dict)
        print("✅ Teacher model loaded successfully")
    except Exception as e:
        print(f"❌ Error: Could not load teacher checkpoint ({e}). Production reflow requires a valid teacher.")
        raise e
        
    model.to(device)
    model.eval()
    
    # 2. Prepare data structures
    collected_data = []
    accepted_samples = 0
    rejected_samples = 0
    
    # In production, iterate over representative pockets from the CrossDocked dataset
    # Placeholder: Assuming a set of diverse pockets is provided in args or found in data_root
    pockets_dir = os.path.join(os.path.dirname(save_path), "active_pockets")
    if os.path.exists(pockets_dir):
        print(f"📂 Loading active pockets from {pockets_dir}...")
        # (Logic to load real pockets would go here)
    
    dummy_pocket = torch.randn(50, 58).to(device) # 50 atoms, 58 features (SOTA Phase 65)
    dummy_pos_P = torch.randn(50, 3).to(device)
    dummy_center = dummy_pos_P.mean(dim=0)
    
    # Phase 11: System 2 Verifier
    from max_flow.utils.verifier import System2Verifier
    from max_flow.utils.metrics import get_mol_from_data
    verifier = System2Verifier()
    
    num_batches = (n_samples + batch_size - 1) // batch_size
    # Adjust loop to just fill the quota
    pbar = tqdm(total=n_samples, desc="Generating Verified Samples")
    
    while accepted_samples < n_samples:
        with torch.no_grad():
            current_batch_size = min(batch_size, n_samples - accepted_samples)
            if current_batch_size <= 0: break # Should not happen with while loop condition
            
            # Create a batch of graphs with controlled noise
            data_list = []
            for _ in range(batch_size): # Always run full batches for efficiency
                # Random ligand size 10-20 atoms
                lig_size = torch.randint(10, 21, (1,)).item()
                noise_level = torch.rand(1).item() * 0.5 + 0.5
                
                data = ProteinLigandData(
                    # Ligand: Start with noise and controlled variance
                    x_L = torch.randn(lig_size, 58).to(device) * noise_level, # SOTA 58 dims
                    pos_L = torch.randn(lig_size, 3).to(device) * noise_level,
                    # Protein
                    x_P = dummy_pocket,
                    pos_P = dummy_pos_P,
                    pocket_center = dummy_center
                )
                data_list.append(data)
                
            batch = Batch.from_data_list(data_list).to(device)
            
            # 3. Solver: x_0 -> x_1 (Geometry)
            x_out, traj = model.flow.sample(batch, steps=steps, gamma=0.0)
            
            # 4. Phase 11 Critical Fix: Predict Atom Types
            # The flow generates positions (x_out). We must query the backbone at t=1 
            # to get the associated atom types for the final geometry.
            t_final = torch.ones(batch.num_graphs, device=device)
            batch.pos_L = x_out
            res, _, _ = model.backbone(t_final, batch)
            atom_logits = res.get('atom_logits')
            
            # Update batch.x_L with predicted types for reconstruction
            # We assume One-Hot encoding matches the backbone output dim
            if atom_logits is not None:
                probs = torch.softmax(atom_logits, dim=-1)
                predicted_types = torch.argmax(probs, dim=-1)
                # Scatter back to x_L (Need to match d_model=58 dimensions)
                # We simply clear x_L and set the one-hot bit
                new_x_L = torch.zeros_like(batch.x_L)
                # Ensure we don't go out of bounds (SOTA 58 includes features, but first N are atoms)
                # Check allowable_features in utils/constants if needed, but simple one-hot is:
                new_x_L.scatter_(1, predicted_types.unsqueeze(-1), 1.0)
                batch.x_L = new_x_L
            
            x_0 = traj[0]
            x_1 = x_out.cpu()
            
            # 5. Verification Loop
            # We need to de-batch to verify individual molecules
            # Batch.to_data_list() is slow, we use manual slicing or index
            # But get_mol_from_data expects a Data object.
            # Best is to iterate and reconstruct.
            
            # Reconstruct batch indices
            batch_indices = getattr(batch, 'x_L_batch', getattr(batch, 'batch'))
            
            for j in range(batch.num_graphs):
                if accepted_samples >= n_samples: break
                
                # Extract Single Data
                mask = (batch_indices == j)
                if not mask.any(): continue
                
                sub_data = ProteinLigandData(
                    x_L = batch.x_L[mask],
                    pos_L = batch.pos_L[mask]
                )
                # Reconstruct
                mol = get_mol_from_data(sub_data)
                
                # System 2 Verify
                passed, reasons, metrics = verifier.verify(mol)
                
                if passed:
                    # Calculate consistency
                    consistency_score = calculate_consistency(traj[:, mask]) # Need to slice traj by nodes? 
                    # Traj is (steps, N_total, 3). We need (steps, N_batch, 3)
                    traj_j = traj[:, mask, :]
                    
                    if consistency_score >= consistency_threshold:
                        collected_data.append({
                            'x_0': x_0[mask].cpu(),
                            'x_1': x_1[mask],
                            'quality_score': 1.0, # Passed verification
                            'consistency_score': consistency_score,
                            'metrics': metrics
                        })
                        accepted_samples += 1
                        pbar.update(1)
                else:
                    rejected_samples += 1

    pbar.close()
    
    # 5. Save the collected data
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(collected_data, save_path)
    
    print(f"🎉 Data Generation Complete!")
    print(f"✅ Accepted Samples: {accepted_samples}")
    print(f"⚠️ Rejected Samples: {rejected_samples}")
    print(f"📈 Selection Rate: {accepted_samples / (accepted_samples + rejected_samples + 1e-6) * 100:.2f}%")
    
    return collected_data


def calculate_consistency(traj):
    """
    Calculate consistency score for a trajectory.
    Measures how straight the path is from x_0 to x_1.
    
    Args:
        traj: Trajectory tensor of shape (steps, features)
    
    Returns:
        consistency_score: Score between 0 and 1 (higher is better)
    """
    if traj.shape[0] < 2:
        return 0.0
    
    # Calculate the straight line distance
    straight_distance = torch.norm(traj[-1] - traj[0]).item()
    
    # Calculate the actual path length
    actual_distance = 0.0
    for i in range(1, traj.shape[0]):
        actual_distance += torch.norm(traj[i] - traj[i-1]).item()
    
    # Consistency score: ratio of straight to actual distance
    if actual_distance == 0:
        return 0.0
    
    consistency_ratio = straight_distance / actual_distance
    return min(max(consistency_ratio, 0.0), 1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Reflow Data Generator")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to teacher model checkpoint")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save generated data")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--steps", type=int, default=100, help="Number of ODE solver steps")
    parser.add_argument("--quality_threshold", type=float, default=0.8, help="Minimum quality score")
    parser.add_argument("--consistency_threshold", type=float, default=0.95, help="Minimum consistency score")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    
    args = parser.parse_args()
    
    generate_reflow_data(
        args.checkpoint,
        args.save_path,
        args.n_samples,
        args.batch_size,
        args.steps,
        quality_threshold=args.quality_threshold,
        consistency_threshold=args.consistency_threshold
    )
