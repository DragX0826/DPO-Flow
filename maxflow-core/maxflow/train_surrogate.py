# maxflow/train_surrogate.py
"""
Phase 3: UARM Surrogate Ensemble Trainer.
Trains multiple GNNProxy models to establish an uncertainty-aware reward frontier.
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from maxflow.models.surrogate import GNNProxy
from maxflow.utils.training import get_optimizer, AverageMeter
from accelerate import Accelerator

def train_ensemble(args):
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    
    # 1. Load Dataset (Expects pairs of {data, scores})
    # For now, we assume a pre-processed .pt dataset is available
    if not os.path.exists(args.data_path):
        accelerator.print(f"❌ Dataset not found at {args.data_path}")
        return

    dataset = torch.load(args.data_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    for m_idx in range(args.num_models):
        accelerator.print(f"\n🔥 Training Ensemble Member {m_idx+1}/{args.num_models}")
        
        # Initialize with different seeds for ensemble diversity
        torch.manual_seed(42 + m_idx)
        model = GNNProxy(hidden_dim=args.hidden_dim)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        
        model, optimizer, loader_m = accelerator.prepare(model, optimizer, loader)
        
        model.train()
        for epoch in range(args.epochs):
            loss_meter = AverageMeter()
            for batch in loader_m:
                optimizer.zero_grad()
                preds = model(batch)
                
                # SOTA Multi-Task Objective
                loss_aff = F.mse_loss(preds['affinity'], batch.affinity)
                loss_qed = F.binary_cross_entropy(preds['qed'], batch.qed)
                loss_sa = F.mse_loss(preds['sa'], batch.sa)
                loss_tpsa = F.mse_loss(preds['tpsa'], batch.tpsa)
                
                loss = loss_aff + loss_qed + loss_sa + 0.1 * loss_tpsa
                
                accelerator.backward(loss)
                optimizer.step()
                loss_meter.update(loss.item())
            
            accelerator.print(f"Epoch {epoch+1} | Loss: {loss_meter.avg:.4f}")
            
        # Save member checkpoint
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            save_path = os.path.join(args.save_dir, f"surrogate_member_{m_idx}.pt")
            torch.save(unwrapped.state_dict(), save_path)
            accelerator.print(f"✅ Saved member to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="checkpoints_surrogate")
    parser.add_argument("--num_models", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    args = parser.parse_args()
    train_ensemble(args)
