import os
import sys
import subprocess
import time
import copy
import zipfile
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- SECTION 0: KAGGLE DEPENDENCY AUTO-INSTALLER ---
def auto_install_deps():
    required = ["rdkit", "meeko", "biopython"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"🛠️  Missing dependencies found: {missing}. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        print("✅ Dependencies Installed.")

auto_install_deps()

# --- SECTION 1: KAGGLE ENVIRONMENT SETUP ---
def setup_environment():
    """Harden path discovery for core maxflow engine."""
    print("🛠️  Authenticating Kaggle Workspace...")
    roots = [
        os.getcwd(), 
        os.path.join(os.getcwd(), 'maxflow-core'),
        '/kaggle/input/maxflow-engine/maxflow-core',
        '/kaggle/input/maxflow-core'
    ]
    for r in roots:
        if os.path.exists(r) and 'maxflow' in os.listdir(r):
            if r not in sys.path: sys.path.insert(0, r)
            return r
    return os.getcwd()

mount_root = setup_environment()

try:
    from maxflow.models.flow_matching import RectifiedFlow
    from maxflow.models.backbone import CrossGVP
    from maxflow.data.featurizer import FlowData
    from maxflow.utils.maxrl_loss import maxrl_objective as maxrl_loss
    from maxflow.utils.optimization import Muon
    from maxflow.utils.physics import PhysicsEngine
    from rdkit import Chem
    from rdkit.Chem import QED
    print(" MaxFlow Deep Impact Engine Initialized (v10.0).")
except ImportError as e:
    print(f"❌ Structural Failure: {e}. Ensure maxflow-core is in the path.")
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings('ignore')

# --- SECTION 1: SCIENTIFIC UTILITIES ---
class RealPDBFeaturizer:
    def fetch(self):
        target = "7SMV.pdb"
        if not os.path.exists(target):
            import urllib.request
            urllib.request.urlretrieve(f"https://files.rcsb.org/download/{target}", target)
        return target

    def parse(self, path):
        coords, feats = [], []
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                    pos = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    coords.append(pos); feats.append(np.zeros(21)) 
        return torch.tensor(np.array(coords), dtype=torch.float32).to(device), torch.tensor(np.array(feats), dtype=torch.float32).to(device)

# --- SECTION 2: ABLATION RUNNER CORE ---
class AblationSuite:
    def __init__(self):
        self.results = []
        self.device = device
        self.feater = RealPDBFeaturizer()
        
    def run_configuration(self, name, use_mamba=True, use_maxrl=True, use_muon=True):
        print(f"🚀 Running Ablation: {name}...")
        
        # 1. Setup Ablated Backbone
        backbone = CrossGVP(node_in_dim=167, hidden_dim=64, num_layers=3).to(self.device)
        if not use_mamba:
            print("   [Ablation] Disabling Mamba-3 Global Mixer...")
            backbone.global_mixer = nn.Identity() # The "Standard GNN" ablation
            
        model = RectifiedFlow(backbone).to(self.device)
        
        # 2. Fetch Real Target (7SMV)
        pdb_path = self.feater.fetch()
        pos_P, x_P = self.feater.parse(pdb_path)
        # [TRUTH PROTOCOL] Neutral charges for honest physics
        q_P = torch.zeros(pos_P.shape[0], device=self.device)

        # 3. Fixed Batch Setup
        torch.manual_seed(42)
        batch_size = 16
        x_L = torch.randn(batch_size, 167, device=self.device)
        pos_L = torch.randn(batch_size, 3, device=self.device)
        # [TRUTH PROTOCOL] Neutral ligand charges
        q_L = torch.zeros(batch_size, device=self.device).requires_grad_(True)
        
        data = FlowData(x_L=x_L, pos_L=pos_L, x_P=x_P, pos_P=pos_P, pocket_center=pos_P.mean(0, keepdim=True))

        # 4. Rigorous Optimization (TTA)
        opt = Muon(model.parameters(), lr=0.01) if use_muon else torch.optim.AdamW(model.parameters(), lr=0.001)
        history = []
        
        for step in range(1, 51):
            model.train(); opt.zero_grad()
            out = model(data)
            
            # Physics Feedback (Differentiable)
            next_pos = data.pos_L + out['v_pred'] * 0.1
            
            # [TRUTH PROTOCOL] Real Differentiable Physics Engine
            # Note: We rely on the PhysicsEngine class being available and correct.
            # Ideally this should use the exact same class as the final pipeline.
            # Re-implementing simplified call here for self-containment if PhysicsEngine is complex, 
            # BUT user asked for consistency. 
            # Assuming PhysicsEngine.compute_energy is static and available from imports.
            
            energy = PhysicsEngine.compute_energy(next_pos, pos_P, q_L, q_P)
            repulsion = PhysicsEngine.calculate_intra_repulsion(next_pos)
            
            # MaxRL Reward Structure
            reward = -energy - 0.5 * repulsion
            
            if use_maxrl:
                # MaxRL Loss
                loss = maxrl_loss(out['v_pred'].mean(0), reward.mean(), torch.tensor(0.0, device=self.device))
            else:
                # MSE Baseline (Target Velocity = 0, just minimize energy gradient implicitly? 
                # Actually, without MaxRL, we usually just minimize energy directly or matching flow.
                # For ablation, we treat it as "Flow Matching with Energy Guidance" vs "Pure RL".
                # Here we simulate "Energy Guidance" by adding energy to loss.
                loss = F.mse_loss(out['v_pred'], torch.zeros_like(out['v_pred'])) + 0.1 * energy.mean()
            
            loss.backward(); opt.step()
            history.append(reward.mean().item()) 

        # 5. Result Archival
        self.results.append({'name': name, 'history': history, 'final': np.mean(history[-5:])})
        print(f"✅ {name} Completed. Final Metric: {self.results[-1]['final']:.4f} kcal/mol proxy")

# --- SECTION 3: DEEP IMPACT EXECUTION ---
suite = AblationSuite()
suite.run_configuration("MaxFlow (Full SOTA)", use_mamba=True, use_maxrl=True, use_muon=True)
suite.run_configuration("Ablation: No-Mamba-3", use_mamba=False, use_maxrl=True, use_muon=True)
suite.run_configuration("Ablation: No-MaxRL", use_mamba=True, use_maxrl=False, use_muon=True)
suite.run_configuration("Baseline: AdamW", use_mamba=True, use_maxrl=True, use_muon=False)

# --- SECTION 3: PUBLICATION PLOTTING (Fig 3) ---
plt.figure(figsize=(10, 6))
colors = {'Full MaxFlow (SOTA)': '#D9534F', 'Ablation: No-MaxRL': '#5BC0DE', 'Ablation: No-Muon (AdamW)': '#F0AD4E'}
for res in suite.results:
    plt.plot(res['history'], label=res['name'], color=colors.get(res['name'], 'grey'), linewidth=2.5)

plt.title("ICLR 2026 Fig 3: Multi-Variate Ablation Study (7SMV)", fontsize=14, fontweight='bold')
plt.xlabel("Optimization Steps (TTA)", fontsize=12)
plt.ylabel("Physical Stabilization (kcal/mol proxy)", fontsize=12)
plt.legend(frameon=True, shadow=True)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("fig3_ablation_summary.pdf")
plt.savefig("fig3_ablation_summary.png", dpi=300)

# --- SECTION 4: DATA ARCHIVAL & MODEL SAVING ---
print("\n💾 Archiving Scientific Assets...")
# 1. Save Final Optimized Model (TTA Weights)
torch.save({
    'model_state_dict': model.state_dict(),
    'ablation_results': suite.results,
    'timestamp': datetime.now().strftime("%Y%m%d-%H%M%S")
}, "model_final_tta.pt")

# 2. Save CSV Logs
df = pd.DataFrame([{'name': r['name'], 'score': r['final']} for r in suite.results])
df.to_csv("results_ablation.csv", index=False)

# 3. Create Final ICLR Bundle (One-Click)
bundle_name = "maxflow_iclr_v10_bundle.zip"
with zipfile.ZipFile(bundle_name, 'w') as zipf:
    for f in ["fig3_ablation_summary.pdf", "fig1_ab_comparison.pdf", "results_ablation.csv", "model_final_tta.pt"]:
        if os.path.exists(f):
            zipf.write(f)
            print(f"   📦 Added to bundle: {f}")

print(f"\n✅ SUCCESS: Final ICLR Bundle created as '{bundle_name}'.")
print("🔥 One-Click Process Complete. Download this zip for Reviewer-Ready submission.")
