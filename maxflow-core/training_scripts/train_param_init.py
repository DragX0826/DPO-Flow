
import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import time

# Adjust path to find 'maxflow' package
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from maxflow.models.backbone import CrossGVP
from maxflow.models.flow_matching import RectifiedFlow
from maxflow.data.featurizer import FlowData

def fast_pretrain():
    print("🚀 Starting Rapid Pre-training (RF Base)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   -> Device: {device}")
    
    # 1. Model Setup
    # Using the same config as inference pipeline
    backbone = CrossGVP(node_in_dim=167, hidden_dim=64, num_layers=3).to(device)
    model = RectifiedFlow(backbone).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    
    print("   -> Dataset: Synthetic (Self-Supervised Structure Learning)")
    
    # 2. Training Loop
    model.train()
    loss_history = []
    steps = 100 # Quick run to get valid weights
    
    t0 = time.time()
    for i in range(steps):
        optimizer.zero_grad()
        
        # Synthetic Batch (simulating drug-protein pairs)
        # Molecules: 20-30 atoms
        # Pockets: 40-50 residues
        batch_size = 16
        
        x_L = torch.randn(batch_size * 25, 167).to(device)
        pos_L = torch.randn(batch_size * 25, 3).to(device)
        x_P = torch.randn(batch_size * 45, 21).to(device)
        pos_P = torch.randn(batch_size * 45, 3).to(device) * 10.0
        
        # Calculate pocket centers (per batch item)
        # For simplicity in this demo, just mean of all pos_P or better per-item
        # Since we are doing batch training, we might need pocket_center to be [B, 3]
        # Let's mock it as [B, 3] random centers
        pocket_center = torch.randn(batch_size, 3).to(device)
        
        # Batch indices
        batch_vec_L = torch.arange(batch_size, device=device).repeat_interleave(25)
        batch_vec_P = torch.arange(batch_size, device=device).repeat_interleave(45)
        
        batch = FlowData(
            x_L=x_L, pos_L=pos_L, x_P=x_P, pos_P=pos_P,
            pocket_center=pocket_center, # Added
            batch=batch_vec_L, x_L_batch=batch_vec_L, x_P_batch=batch_vec_P
        )
        
        # Flow Matching Loss
        # Create noise and time
        t = torch.rand(batch_size, device=device) # [B]
        t_batch = t.repeat_interleave(25) # [N_atoms]
        
        z = torch.randn_like(x_L)
        x_L_t = (1 - t_batch.unsqueeze(-1)) * z + t_batch.unsqueeze(-1) * x_L
        
        # In a real script we'd call model.get_loss, but here we manually call backbone for speed/control
        # Or if RectifiedFlow has loss method
        # Let's assume model.loss() exists or we do manual
        
        # Manual FM Loss for robust demo
        # Interpolated state
        batch.x_L = x_L_t 
        
        # Forward
        out, _, _ = model.backbone(t_batch, batch, return_latent=False)
        v_pred = out['v_pred'] if 'v_pred' in out else out['x'] # Adjust key
        
        # Target: x_1 - x_0 = x_L - z
        v_target = x_L - z
        
        loss = F.mse_loss(v_pred, v_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        loss_history.append(loss.item())
        if i % 10 == 0:
            print(f"   Step {i}: Loss = {loss.item():.4f}")
            
    print(f"✅ Pre-training Complete ({time.time()-t0:.1f}s). Final Loss: {loss_history[-1]:.4f}")
    
    # 3. Save
    save_path = r"d:\Drug\kaggle_submission\maxflow-core\checkpoints\maxflow_pretrained.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {'hidden_dim': 64, 'num_layers': 3},
        'step': steps
    }, save_path)
    print(f"💾 Checkpoint Saved: {save_path}")

if __name__ == "__main__":
    fast_pretrain()
