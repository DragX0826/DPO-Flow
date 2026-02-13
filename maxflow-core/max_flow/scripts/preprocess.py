# max_flow/scripts/preprocess.py

import os
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import json
import argparse
from max_flow.data.featurizer import ProteinLigandFeaturizer
from max_flow.utils.pca import compute_pca_rotation, apply_canonicalization

def process_item(item, data_root, save_root, featurizer):
    pdb_rel, sdf_rel = item
    pdb_path = os.path.join(data_root, pdb_rel)
    sdf_path = os.path.join(data_root, sdf_rel)
    
    save_name = pdb_rel.replace('/', '_').replace('\\', '_') + ".pt"
    save_path = os.path.join(save_root, save_name)
    
    if os.path.exists(save_path):
        return save_name
    
    try:
        data = featurizer(pdb_path, sdf_path)
        if data is None: return None
        
        # --- HALO: PCA Canonicalization ---
        # Align to Principal Axes before saving to disk
        rot, center = compute_pca_rotation(data.pos_P)
        
        data.pos_P = apply_canonicalization(data.pos_P, torch.zeros(data.pos_P.size(0), dtype=torch.long), rot, center)
        data.pos_L = apply_canonicalization(data.pos_L, torch.zeros(data.pos_L.size(0), dtype=torch.long), rot, center)
        
        # Save center and rot for reconstruction if needed
        data.pca_rot = rot
        data.pca_center = center
        
        torch.save(data, save_path)
        return save_name
    except Exception as e:
        # print(f"Error processing {pdb_rel}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Dataset Hardening: PCA Pre-alignment")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--save_root", type=str, default="data/processed_pca")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    
    os.makedirs(args.save_root, exist_ok=True)
    
    with open(args.index, 'r') as f:
        items = json.load(f)
        
    featurizer = ProteinLigandFeaturizer()
    
    print(f"Starting Preprocessing of {len(items)} items using {args.num_workers} workers...")
    
    # We use a simple loop for Windows compatibility in this demo, 
    # but in a real Scaling phase, we would use mp.Pool
    processed_index = []
    
    pbar = tqdm(items)
    for item in pbar:
        res = process_item(item, args.data_root, args.save_root, featurizer)
        if res:
            processed_index.append(res)
            
    # Save the new index (direct to .pt files)
    with open(os.path.join(args.save_root, "processed_index.json"), "w") as f:
        json.dump(processed_index, f, indent=4)
        
    print(f"Done! Processed {len(processed_index)} items.")

if __name__ == "__main__":
    main()
