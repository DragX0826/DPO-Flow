# maxflow/scripts/evaluate_sota.py
import os
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from maxflow.models.backbone import CrossGVP
from maxflow.models.flow_matching import RectifiedFlow
from maxflow.data.dataset import LazyDockingDataset, collate_fn
from maxflow.utils.metrics import MultiObjectiveScorer
from torch.utils.data import DataLoader

def evaluate_sota(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing SOTA Evaluation | Device: {device}")
    
    # 1. Load Flagship Model (1.26M Parameters)
    # node_in_dim = 167 (Flagship standard)
    backbone = CrossGVP(
        node_in_dim=167, 
        hidden_dim=args.hidden_dim, 
        num_layers=args.num_layers
    )
    model = RectifiedFlow(backbone).to(device)
    
    print(f"📂 Loading Checkpoint: {args.ckpt}")
    state_dict = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 2. Scorer Initialization (Parallel RDKit)
    scorer = MultiObjectiveScorer()
    
    # 3. Data Loading
    if args.target_pdb and os.path.exists(args.target_pdb):
        print(f"🎯 Target Mode: Evaluating specifically on {args.target_pdb}")
        from maxflow.data.featurizer import ProteinLigandFeaturizer
        featurizer = ProteinLigandFeaturizer()
        # Mock ligand positions for sampling seed
        mock_ligand_pos = torch.zeros((1, 3)) 
        data_list = [featurizer.featurize(args.target_pdb, None, ligand_pos=mock_ligand_pos)]
        dataset = data_list
    else:
        # Discover shards in data_root
        import glob
        shards = sorted(glob.glob(os.path.join(args.data_root, "*.pt")))
        if not shards:
            raise FileNotFoundError(f"No .pt shards found in {args.data_root}")
        
        # Load first shard for evaluation (Cross-Docked test set shards)
        print(f"📦 Loading evaluation data from {shards[0]}...")
        shard_data = torch.load(shards[0], map_location="cpu", weights_only=False)
        
        # Handle different shard formats (Raw list of Data vs Dict with Index)
        if isinstance(shard_data, list):
            # If it's a list of Data objects, use it directly (pre-featurized)
            print(f"✨ Detected pre-featurized shard (List of Data).")
            dataset = shard_data
            if args.num_samples < len(dataset):
                dataset = dataset[:args.num_samples]
        elif isinstance(shard_data, dict) and 'index' in shard_data:
            # If it's an index mapping, use LazyDockingDataset
            print(f"🗂️ Detected index mapping shard. Using LazyDockingDataset.")
            index_mapping = shard_data['index']
            if args.num_samples < len(index_mapping):
                index_mapping = index_mapping[:args.num_samples]
            dataset = LazyDockingDataset(index_mapping, root_dir=args.data_root)
        else:
            # Fallback for other formats
            dataset = shard_data
            if args.num_samples < len(dataset):
                dataset = dataset[:args.num_samples]
            
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    results = []
    print(f"🔬 Starting Inference on {len(dataset)} samples...")
    
    for batch in tqdm(dataloader):
        if batch is None: continue
        batch = batch.to(device)
        
        import time
        start_t = time.time()
        with torch.no_grad():
            # Flagship Sampling (ODE-like steps)
            pos_gen, _ = model.sample(batch, steps=args.steps)
            
            # Update batch with generated positions for scoring
            batch.pos_L = pos_gen
            
            # Multi-Objective Scoring (Vina, QED, SA, Clashes)
            rewards, valid_mask = scorer.calculate_batch_reward(batch)
        
        elapsed = (time.time() - start_t) / batch.num_graphs
            
        # Memory Management: Clear cache after each batch to prevent OOM
        if device.type == "cuda":
            torch.cuda.empty_cache()
            
        # Process Results
        for i in range(batch.num_graphs):
            if valid_mask[i]:
                res = {
                    "reward": rewards[i].item(),
                    "valid": True,
                    "time": elapsed
                }
                results.append(res)
                
                # SOTA Addition: Save Top Molecules as structures
                # We save the RDKit mol object from the scorer if available
                mol = getattr(scorer, 'last_mols', [None]*batch.num_graphs)[i]
                if mol is not None:
                    mol.SetProp("_Name", f"Sample_{len(results)}_{rewards[i].item():.3f}")
                    top_mols_dir = os.path.join(args.output_dir, "top_molecules")
                    os.makedirs(top_mols_dir, exist_ok=True)
                    from rdkit import Chem
                    Chem.MolToMolFile(mol, os.path.join(top_mols_dir, f"mol_{len(results)}.sdf"))
            else:
                results.append({"reward": float('nan'), "valid": False})
                
    # 4. Save Results
    df = pd.DataFrame(results)
    out_path = os.path.join(args.output_dir, "benchmark_results.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(out_path, index=False)
    
    valid_count = df['valid'].sum()
    print(f"\n✅ Evaluation Complete.")
    print(f"📊 Validity Rate: {valid_count/len(df):.2%}")
    print(f"💾 Results saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MaxFlow SOTA Evaluation")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data_root", type=str, required=True, help="Path to CrossDocked shards")
    parser.add_argument("--target_pdb", type=str, default=None, help="Specific target PDB for evaluation")
    parser.add_argument("--output_dir", type=str, default="./results/eval", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10, help="Sampling steps")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    args = parser.parse_args()
    
    evaluate_sota(args)
