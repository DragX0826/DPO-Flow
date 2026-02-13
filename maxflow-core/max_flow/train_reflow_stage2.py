# max_flow/train_reflow_stage2.py

import os
import torch
import argparse
from torch.utils.data import DataLoader
from max_flow.data.dataset import LazyDockingDataset, collate_fn
from max_flow.models.backbone import CrossGVP
from max_flow.models.flow_matching import RectifiedFlow
from tqdm import tqdm

def train_stage2(args):
    """
    Stage 2 Reflow: Training on (x_0, x_1_pred) to straighten trajectories.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Stage 1 Model (Teacher)
    backbone_1 = CrossGVP(node_in_dim=161, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    teacher = RectifiedFlow(backbone_1).to(device)
    teacher.load_state_dict(torch.load(args.teacher_path, map_location=device))
    teacher.eval()
    
    # 2. Setup Stage 2 Model (Student)
    backbone_2 = CrossGVP(node_in_dim=161, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    student = RectifiedFlow(backbone_2).to(device)
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    
    # 3. Data
    index_mapping = [("mock_pdb.pdb", "mock_sdf.sdf")] * 100 # Placeholder
    dataset = LazyDockingDataset(index_mapping, root_dir=args.data_root)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    print("Starting Reflow Stage 2: Trajectory Straightening...")
    for epoch in range(args.epochs):
        student.train()
        pbar = tqdm(dataloader, desc=f"Stage 2 Epoch {epoch+1}")
        for batch in pbar:
            if batch is None: continue
            batch = batch.to(device)
            
            # --- The Reflow Logic ---
            # 1. Generate x_1_pred using Stage 1 Model (1-step Euler)
            with torch.no_grad():
                # Note: We need x_0 to be consistent. 
                # We reuse the logic from rf.loss or rf.sample
                # For Reflow, we train on (x_0, x_1_student) where x_1_student = teacher.sample(x_0)
                x_1_teacher, _ = teacher.sample(batch, steps=1)
            
            # 2. Update batch.pos_L to be the "straightened" target
            batch.pos_L = x_1_teacher
            
            # 3. Train student to match this "straight" velocity field
            optimizer.zero_grad()
            loss = student.loss(batch)
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(loss=loss.item())
            
        # Save Student
        os.makedirs("checkpoints_reflow", exist_ok=True)
        torch.save(student.state_dict(), f"checkpoints_reflow/reflow_v2_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_path", type=str, required=True, help="Path to Stage 1 checkpoint")
    parser.add_argument("--data_root", type=str, default="smoke_test_data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    train_stage2(args)
