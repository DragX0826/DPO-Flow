# maxflow/scripts/analyze_trajectory.py

import torch
import argparse
import numpy as np
from maxflow.models.backbone import CrossGVP
from maxflow.models.flow_matching import RectifiedFlow
from maxflow.data.dataset import LazyDockingDataset, collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm

def compute_curvature(model, batch, device):
    """
    Measures how much the predicted vector field deviates from a straight line.
    Curvature = 1 - CosineSimilarity(v_pred, x_1 - x_0)
    """
    model.eval()
    with torch.no_grad():
        # Setup consistent noise x_0
        center = batch.pocket_center
        noise = torch.randn_like(batch.pos_L)
        x_0 = noise + center
        x_1 = batch.pos_L
        
        # Test at mid-point t=0.5 (most critical for curvature)
        t = torch.tensor([0.5] * batch.num_graphs, device=device)
        
        # Predicted velocity
        batch.pos_L = 0.5 * x_1 + 0.5 * x_0 # x_t at t=0.5
        v_pred = model.backbone(t, batch)
        
        # Ideal velocity (straight line)
        v_ideal = x_1 - x_0
        
        # Cosine Similarity
        cos_sim = torch.nn.functional.cosine_similarity(v_pred, v_ideal, dim=-1)
        curvature = 1.0 - cos_sim.mean().item()
        
    return curvature

def run_analysis(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    backbone = CrossGVP(node_in_dim=161, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    model = RectifiedFlow(backbone).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    
    # Data
    dataset = LazyDockingDataset([("mock_pdb.pdb", "mock_sdf.sdf")]*20, root_dir=args.data_root)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    
    curvatures = []
    print(f"Analyzing trajectory straightness for: {args.ckpt}")
    for batch in tqdm(dataloader):
        if batch is None: continue
        batch = batch.to(device)
        c = compute_curvature(model, batch, device)
        curvatures.append(c)
        
    avg_c = np.mean(curvatures)
    print(f"\n--- Mechanistic Results ---")
    print(f"Average Trajectory Curvature: {avg_c:.6f}")
    if avg_c < 0.1:
        print("Status: HIGHLY STRAIGHT (Ideal for 1-Step generation)")
    else:
        print("Status: CURVED (Suggests more Reflow or higher N-steps needed)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="smoke_test_data")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    args = parser.parse_args()
    
    run_analysis(args)
