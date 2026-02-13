"""
Reflow Data Generator (Teacher Rollout)
Phase 25: Single-Step Distillation

Generates (Noise, Data) pairs from a pre-trained Teacher model (Phase 24).
These pairs are used to train the Student model to jump from x_0 to x_1 in a single step.

Strategy:
1. Load Phase 24 Teacher Model.
2. Generate latent noise x_0.
3. Solve ODE to get x_1 (using high-precision solver, e.g., Euler 100 steps).
4. Save pairs (x_0, x_1) and conditioning (pocket) for distillation.
"""

import torch
import os
import argparse
from tqdm import tqdm
from torch_geometric.data import Batch
from maxflow.models.max_rl import MaxFlow
from maxflow.utils.chem import ProteinLigandData

def generate_reflow_data(
    checkpoint_path, 
    save_path, 
    n_samples=1000, 
    batch_size=32, 
    steps=100,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    print(f"🚀 Reflow Teacher Rollout")
    print(f"Teacher: {checkpoint_path}")
    print(f"Target: {save_path}")
    print(f"Samples: {n_samples} | Solver Steps: {steps}")
    
    # 1. Load Teacher
    # Mock config for loading (should be loaded from args/config in real app)
    # Using a minimal compatible config
    try:
        model = MaxFlow()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Teacher model loaded successfully")
    except Exception as e:
        print(f"⚠️ Could not load checkpoint directly ({e}). Initializing random teacher for testing.")
        model = MaxFlow()
        
    model.to(device)
    model.eval()
    
    # 2. Mock Data Source (In real usage, this should iterate over validation set pockets)
    # We create a dummy pocket for demonstration
    dummy_pocket = torch.randn(50, 6).to(device) # 50 atoms, 6 features
    dummy_pos_P = torch.randn(50, 3).to(device)
    dummy_center = dummy_pos_P.mean(dim=0)
    
    collected_data = []
    
    num_batches = (n_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            current_batch_size = min(batch_size, n_samples - i * batch_size)
            
            # Create a batch of graphs
            data_list = []
            for _ in range(current_batch_size):
                # Random ligand size 10-20 atoms
                lig_size = torch.randint(10, 21, (1,)).item()
                
                data = ProteinLigandData(
                    # Ligand (Noise init handled by sampler, but we need structure)
                    x_L = torch.zeros(lig_size, 6).to(device), # Dummy features
                    pos_L = torch.randn(lig_size, 3).to(device), # Placeholder
                    
                    # Protein (Repeated)
                    x_P = dummy_pocket,
                    pos_P = dummy_pos_P,
                    pocket_center = dummy_center
                )
                data_list.append(data)
                
            batch = Batch.from_data_list(data_list).to(device)
            
            # 3. Solver: x_0 -> x_1
            # We need to capture x_0 (noise) and x_1 (result)
            # The sample method in flow_matching usually generates its own noise.
            # We might need to modify or hook into it, or just use the noise it generates if returned.
            # Our modified sample method in flow_matching.py returns (x_final, traj).
            # Traj[0] is x_0.
            
            x_out, traj = model.flow.sample(batch, steps=steps, gamma=0.0) # No guidance for reflow base
            
            x_0 = traj[0]
            x_1 = x_out.cpu()
            
            # 4. Save Logic
            # Fix: Save full Graph objects to preserve edges and features for GNN
            
            # Split back to individual graphs
            idx_slices = [0] + torch.cumsum(batch.batch.bincount(), 0).tolist()
            
            for j in range(current_batch_size):
                start, end = idx_slices[j], idx_slices[j+1]
                
                # Clone the original data object to preserve structure (x_L, x_P, edges etc.)
                # Note: data_list[j] matches the i-th batch's j-th item
                graph_data = data_list[j].clone()
                
                # Update positions
                # x_0 (Noise) -> becomes input pos_L
                graph_data.pos_L = x_0[start:end].cpu()
                
                # x_1 (Teacher Output) -> becomes target
                # We store it as a new attribute 'target_pos'
                graph_data.target_pos = x_1[start:end].cpu()
                
                # Remove batch attribute if present (will be re-batched by DataLoader)
                if hasattr(graph_data, 'batch'):
                    del graph_data.batch
                if hasattr(graph_data, 'ptr'):
                     del graph_data.ptr
                
                collected_data.append(graph_data)
                
    # 5. Save Dataset
    torch.save(collected_data, save_path)
    print(f"🎉 Saved {len(collected_data)} full graph pairs to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/MaxRL_model_epoch_5.pt")
    parser.add_argument("--save_path", type=str, default="data/reflow_dataset.pt")
    parser.add_argument("--n_samples", type=int, default=100)
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    generate_reflow_data(args.checkpoint, args.save_path, args.n_samples)
