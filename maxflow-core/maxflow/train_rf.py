# maxflow/train_rf.py

import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from maxflow.data.dataset import LazyDockingDataset, collate_fn
from maxflow.models.backbone import CrossGVP
from maxflow.models.flow_matching import RectifiedFlow
from maxflow.utils.training import (
    optimize_for_intel, get_optimizer, get_scheduler, 
    AverageMeter, CSVLogger, SilentStepLogger
)
import warnings
# Suppress Pydantic v2 metadata warnings from underlying libraries
warnings.filterwarnings("ignore", module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
from accelerate import Accelerator

def train(args):
    # 0. Initialize Accelerator
    # Mixed Precision: SOTA preference for bfloat16 if T4/A100/H100
    mixed_precision = "fp16" # Default
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        mixed_precision = "bf16"

    accelerator = Accelerator(mixed_precision=mixed_precision)
    device = accelerator.device
    accelerator.print(f"Training (RF) on device: {device} | Precision: {mixed_precision}")
    
    # 1. Dataset & DataLoader (Sharded Loading Strategy)
    manifest_path = os.path.join(args.data_root, "shards_manifest.json")
    manifest = []
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        accelerator.print(f"Loading Sharded Dataset: {len(manifest)} shards found.")
    
    if not manifest:
        # Final Fallback: Direct scan
        import glob
        shards = glob.glob(os.path.join(args.data_root, "**/*.pt"), recursive=True)
        if shards:
            accelerator.print(f"⚠️ Manifest empty. Found {len(shards)} shards via scan.")
            start_idx = 0
            for s in sorted(shards):
                rel_path = os.path.relpath(s, args.data_root)
                manifest.append({"file": rel_path, "start_idx": 0, "end_idx": 1}) # Simple fallback
        else:
            raise ValueError(f"❌ No training data (.pt or shards_manifest.json) found in {args.data_root}")

    # Load all graphs from shards into a flat list
    from torch_geometric.data import Data as PyGData
    KEEP_FIELDS = {'x_L', 'pos_L', 'atom_types', 'edge_index_L', 'x_P', 'pos_P', 'normals_P', 
                   'pocket_center', 'num_nodes_L', 'num_nodes_P', 'pos_metals'}
    
    all_graphs = []
    for shard_info in manifest:
        shard_path = os.path.join(args.data_root, shard_info["file"])
        shard_data = torch.load(shard_path, weights_only=False)
        if not isinstance(shard_data, list):
            shard_data = [shard_data]
        for d in shard_data:
            # Rebuild clean Data without synthetic motif fields
            clean = PyGData()
            for key in KEEP_FIELDS:
                val = getattr(d, key, None)
                if val is not None:
                    setattr(clean, key, val)
            clean.num_nodes = clean.x_L.size(0) + clean.x_P.size(0)
            all_graphs.append(clean)
    
    accelerator.print(f"  Total graphs loaded: {len(all_graphs)}")
    
    from torch.utils.data import Dataset as TorchDataset
    
    class ShardedGraphDataset(TorchDataset):
        def __init__(self, data_list):
            self.data_list = data_list
        def __len__(self):
            return len(self.data_list)
        def __getitem__(self, idx):
            data = self.data_list[idx]
            # PyG needs num_nodes for Batch.from_data_list with custom edge_index names
            if not hasattr(data, 'num_nodes') or data.num_nodes is None:
                data.num_nodes = data.x_L.size(0) + data.x_P.size(0)
            # Remove synthetic motif assignments — they cause NaN in motif pooling.
            # The atom-level velocity path is more stable for LMDB-converted data.
            if hasattr(data, 'atom_to_motif'):
                del data.atom_to_motif
            if hasattr(data, 'joint_indices'):
                del data.joint_indices
            if hasattr(data, 'num_motifs'):
                del data.num_motifs
            return data
    
    dataset = ShardedGraphDataset(all_graphs)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0  # In-memory dataset, no need for workers
    )
    
    # 2. Model
    # Updated node_in_dim=58 for Organic Hardening (Phase 65)
    backbone = CrossGVP(node_in_dim=58, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    flow_model = RectifiedFlow(backbone, use_dmd=args.use_dmd)
    
    if hasattr(torch, "compile"):
        accelerator.print("Compiling backbone with torch.compile...")
        flow_model.backbone = torch.compile(flow_model.backbone, mode="reduce-overhead")
    
    # 3. Optimizer
    optimizer = get_optimizer(flow_model, learning_rate=args.lr)
    
    # 4. Prepare with Accelerator
    flow_model, optimizer, dataloader = accelerator.prepare(flow_model, optimizer, dataloader)
    
    # 5. Scheduler
    total_steps = len(dataloader) * args.epochs
    scheduler = get_scheduler(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    scheduler = accelerator.prepare(scheduler)
    
    # 6. Training Loop Setup
    if accelerator.is_main_process:
        csv_logger = CSVLogger("rf_training_stats.csv", ["epoch", "step", "loss", "skip", "lr"])

    flow_model.train()
    for epoch in range(args.epochs):
        loss_meter = AverageMeter()
        skip_count = 0
        
        # Silent Logger for terminal
        step_logger = SilentStepLogger(
            accelerator, 
            total_steps=len(dataloader), 
            interval=50, 
            desc=f"RF Epoch {epoch+1}"
        )
        
        for i, batch in enumerate(dataloader):
            if batch is None: continue
            
            optimizer.zero_grad()
            
            # Forward & Loss
            loss = flow_model.loss(batch)
            
            # Skip NaN batches with diagnostic output
            if torch.isnan(loss) or torch.isinf(loss):
                skip_count += 1
                if skip_count <= 5 or skip_count % 100 == 0:
                    accelerator.print(f"\n⚠️ NaN detected in batch! Diagnostic Stats:")
                    accelerator.print(f"  - Pocket Center: max={batch.pocket_center.abs().max():.4f}, mean={batch.pocket_center.mean():.4f}")
                    accelerator.print(f"  - Ligand X: max={batch.x_L.abs().max():.4f}, mean={batch.x_L.mean():.4f}")
                    accelerator.print(f"  - Protein Pos: max={batch.pos_P.abs().max():.4f}, mean={batch.pos_P.mean():.4f}")
                continue
            
            # Accelerator Backward
            accelerator.backward(loss)
            
            # Gradient clipping (crucial for VIB stability)
            torch.nn.utils.clip_grad_norm_(flow_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            loss_meter.update(loss.item(), batch.num_graphs)
            
            # Silent Step Logging
            if accelerator.is_local_main_process:
                current_lr = optimizer.param_groups[0]['lr']
                step_logger.log(i + 1, {"loss": loss_meter.val, "avg_loss": loss_meter.avg, "skip": skip_count})
                
                # Persistence
                if accelerator.is_main_process:
                    csv_logger.log({
                        "epoch": epoch + 1,
                        "step": i + 1,
                        "loss": loss.item(),
                        "skip": skip_count,
                        "lr": current_lr
                    })
            
        # Save checkpoint
        if accelerator.is_main_process:
            os.makedirs("checkpoints", exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(flow_model)
            # Epoch-specific and generic "last" checkpoint for pipeline compatibility
            torch.save(unwrapped_model.state_dict(), f"checkpoints/rf_model_epoch_{epoch+1}.pt")
            torch.save(unwrapped_model.state_dict(), "checkpoints/rf_last.pt")
            accelerator.print(f"Epoch {epoch+1} Complete. Avg Loss: {loss_meter.avg:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MaxFlow Rectified Flow Training")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_dmd", action="store_true", help="Enable DMD Diversity Loss")
    
    args = parser.parse_args()
    train(args)
