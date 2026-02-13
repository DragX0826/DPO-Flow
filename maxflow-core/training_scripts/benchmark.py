import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.data import Batch

from maxflow.models.backbone import CrossGVP
from maxflow.models.flow_matching import RectifiedFlow
from maxflow.utils.metrics import get_mol_from_data, MultiObjectiveScorer
from maxflow.data.featurizer import FlowData
from maxflow.data.dataset import collate_fn

def load_universal_engine(checkpoint_path, device='cuda'):
    """
    Loads the pre-trained Universal Geometric Drug Design Engine.
    """
    print(f"Loading Universal Engine from {checkpoint_path}...")
    
    # Initialize Architecture (Standard Configuration)
    # Note: Ensure these hyperparameters match training!
    backbone = CrossGVP(node_in_dim=167, hidden_dim=64, num_layers=3)
    # Enable DMD and Physics by default for Inference
    model = RectifiedFlow(backbone, use_dmd=True) 
    
    # Load Weights
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False) # Allow missing keys for potential head updates
    model.to(device)
    model.eval()
    
    print("Done: Engine Loaded & Ready for Zero-Shot Inference.")
    return model

def run_zero_shot_generation(model, target_list, num_samples=50, device='cuda'):
    """
    Generates candidates for a list of unseen targets.
    
    target_list: List of dicts {'name': str, 'pocket_center': (x,y,z), 'dummy_pocket': Tensor}
    """
    results = []
    scorer = MultiObjectiveScorer()
    
    print(f"Starting Zero-Shot Generalization Test on {len(target_list)} Targets...")
    
    with torch.no_grad():
        for target in target_list:
            name = target['name']
            center = torch.tensor(target['pocket_center'], dtype=torch.float32, device=device)
            # Dummy pocket (In real usage, this would be computed from PDB)
            # using 'dummy_pocket' tensor from input or creating random features
            x_P = target.get('x_P', torch.randn(50, 21, device=device)) 
            pos_P = target.get('pos_P', torch.randn(50, 3, device=device) + center)
            
            print(f"\n🧪 Target: {name} (Zero-Shot)")
            print(f"   Pocket Center: {center.cpu().numpy()}")
            
            generated_mols = []
            valid_count = 0
            
            # Batch Generation
            batch_size = 10
            for _ in tqdm(range(0, num_samples, batch_size), desc="Generating"):
                current_bs = min(batch_size, num_samples - len(generated_mols))
                if current_bs <= 0: break
                
                data_list = []
                data_list = []
                for _ in range(current_bs):
                    # Random Size 15-25 atoms for diverse exploration
                    lig_size = torch.randint(15, 26, (1,)).item()
                    data = FlowData(
                        x_L = torch.randn(lig_size, 167).to(device),
                        pos_L = torch.randn(lig_size, 3).to(device) + center,
                        num_nodes_L = torch.tensor([lig_size], dtype=torch.long).to(device),
                        x_P = x_P,
                        pos_P = pos_P,
                        num_nodes_P = torch.tensor([x_P.size(0)], dtype=torch.long).to(device),
                        pocket_center = center.view(1, 3)
                    )
                    data_list.append(data)
                
                batch = collate_fn(data_list)
                if batch is None: continue
                batch = batch.to(device)
                
                # SOTA Inference: High precision sampling
                # Gamma=2.0 for strong guidance on unseen targets
                x_out, traj = model.sample(batch, steps=20, gamma=2.0)
                
                # Reconstruct
                # Predict Atom Types first (Critical Step)
                t_final = torch.ones(batch.num_graphs, device=device)
                batch.pos_L = x_out
                res, _, _ = model.backbone(t_final, batch)
                atom_logits = res.get('atom_logits')
                
                if atom_logits is not None:
                     probs = torch.softmax(atom_logits, dim=-1)
                     types = torch.argmax(probs, dim=-1)
                     batch.x_L = torch.zeros_like(batch.x_L)
                     batch.x_L.scatter_(1, types.unsqueeze(-1), 1.0)

                # De-batch
                batch_idx = getattr(batch, 'x_L_batch', getattr(batch, 'batch'))
                for j in range(current_bs):
                    mask = (batch_idx == j)
                    if not mask.any(): continue
                    
                    sub_data = FlowData(x_L=batch.x_L[mask], pos_L=batch.pos_L[mask])
                    mol = get_mol_from_data(sub_data)
                    
                    if mol is not None:
                         # Verify Validity
                         try:
                             Chem.SanitizeMol(mol)
                             generated_mols.append(mol)
                             valid_count += 1
                         except:
                             pass
            
            # Evaluate
            print(f"   Validity Rate: {valid_count}/{num_samples} ({valid_count/num_samples:.1%})")
            
            if generated_mols:
                 # Calculate Diversity (Tanimoto)
                 from rdkit import DataStructs
                 from rdkit.Chem import AllChem
                 fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2) for m in generated_mols]
                 if len(fps) > 1:
                     divs = []
                     for i in range(len(fps)):
                         for k in range(i+1, len(fps)):
                             divs.append(1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[k]))
                     diversity = np.mean(divs)
                 else:
                     diversity = 0.0
                     
                 # QED
                 qeds = [scorer.calculate_qed(m) for m in generated_mols]
                 avg_qed = np.mean(qeds)
                 
                 print(f"   🌈 Diversity: {diversity:.3f}")
                 print(f"   💊 Avg QED: {avg_qed:.3f}")
                 
                 results.append({
                     'target': name,
                     'validity': valid_count/num_samples,
                     'diversity': diversity,
                     'qed': avg_qed,
                     'samples': generated_mols
                 })
            
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/rf_last.pt")
    parser.add_argument("--targets_json", type=str, help="Path to JSON list of targets")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Engine
    try:
        model = load_universal_engine(args.checkpoint, device)
    except Exception as e:
        print(f"Could not load model: {e}")
        # Build dummy for CI/Demo
        print("Warning: Running in Demo Mode with randomized architecture")
        backbone = CrossGVP(node_in_dim=58, hidden_dim=64, num_layers=3)
        model = RectifiedFlow(backbone, use_dmd=True).to(device)
    
    # 2. Define Targets (Simulation of Cross-Species)
    targets = []
    if args.targets_json:
        import json
        with open(args.targets_json, 'r') as f:
            targets = json.load(f)
    else:
        # Defaults: Simulate 3 distinct binding sites
        print("Warning: No targets provided. Using Simulated Testbed (Cross-Species Proxies).")
        targets = [
            {'name': 'MERS-CoV (Simulated)', 'pocket_center': (10.0, 10.0, 10.0)},
            {'name': 'Zika NS2B-NS3 (Simulated)', 'pocket_center': (-5.0, 20.0, 5.0)},
            {'name': 'Human Cathepsin L (Simulated)', 'pocket_center': (0.0, 0.0, 0.0)}
        ]
    
    # 3. Run Test
    results = run_zero_shot_generation(model, targets, device=device)
    
    # 4. Report
    print("\n" + "="*50)
    print("🏆 UNIVERSAL ENGINE: SOTA GENERALIZATION REPORT")
    print("="*50)
    df = pd.DataFrame(results)
    if not df.empty:
        print(df[['target', 'validity', 'diversity', 'qed']].to_markdown())
        
        avg_validity = df['validity'].mean()
        avg_diversity = df['diversity'].mean()
        
        if avg_validity > 0.1 and avg_diversity > 0.6:
            print("\nDone: Universal Engine demonstrates Zero-Shot Viability.")
        else:
            print("\nWarning: Low generalization score. Further Reflow needed.")
    else:
        print("No valid results generated.")
    
    # Save Results
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/zero_shot_report.csv")
    print("\n📄 Detailed report saved to results/zero_shot_report.csv")
