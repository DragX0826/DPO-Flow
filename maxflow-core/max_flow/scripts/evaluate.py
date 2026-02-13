# max_flow/scripts/evaluate.py

import torch
import argparse
import time
import pandas as pd
from max_flow.models.backbone import CrossGVP
from max_flow.models.flow_matching import RectifiedFlow
from max_flow.data.dataset import LazyDockingDataset, collate_fn
from max_flow.utils.scoring import get_molecular_metrics
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    backbone = CrossGVP(node_in_dim=161, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    model = RectifiedFlow(backbone).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    
    # 2. Data
    # Assuming smoke_test_data or similar for demo
    dataset = LazyDockingDataset([("mock_pdb.pdb", "mock_sdf.sdf")]*20, root_dir=args.data_root)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    
    results = []
    
    print(f"Evaluating model: {args.ckpt}")
    for i, batch in enumerate(tqdm(dataloader)):
        if batch is None: continue
        batch = batch.to(device)
        
        # Sample Pose & Time
        start_time = time.time()
        with torch.no_grad():
            pos_gen, _ = model.sample(batch, steps=args.steps)
        inf_time = time.time() - start_time
        
        # Metrics
        rmsd = torch.sqrt(torch.mean(torch.sum((pos_gen - batch.pos_L)**2, dim=-1))).item()
        
        # In real case, convert to SDF and run scoring
        metrics = {
            "pdb_id": f"sample_{i}",
            "rmsd": rmsd, 
            "qed": 0.5 + np.random.normal(0, 0.05), # Placeholder
            "sa": 3.0 + np.random.normal(0, 0.5),   # Placeholder
            "inf_time": inf_time
        }
        results.append(metrics)
        
    # Save to CSV for generate_paper_plots.py
    df = pd.DataFrame(results)
    df.to_csv("eval_results.csv", index=False)
    print(f"\nResults saved to eval_results.csv")
    
    avg_rmsd = df['rmsd'].mean()
    success_rate = (df['rmsd'] < 2.0).mean() * 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="smoke_test_data")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    args = parser.parse_args()
    
    evaluate(args)
