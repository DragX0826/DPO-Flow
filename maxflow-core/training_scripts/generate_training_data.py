#!/usr/bin/env python
"""
MaxFlow: Synthetic Training Data Generator.
Generates structurally valid protein-ligand graph data for training
when real CrossDocked2020 shards are unavailable.

The generated graphs have identical schema to real featurized data,
ensuring full compatibility with the training pipeline.
"""

import os
import json
import argparse
import torch
import numpy as np
from torch_geometric.data import Data

def generate_ligand(num_atoms=None):
    """Generate a synthetic ligand with realistic atom counts."""
    if num_atoms is None:
        num_atoms = np.random.randint(8, 45)  # Drug-like range
    
    # x_L: atom features (167-dim to match Pharmacophore featurizer)
    x_L = torch.zeros(num_atoms, 167)
    # One-hot atom type in first 10 dims (C, N, O, S, F, Cl, Br, P, I, Other)
    for i in range(num_atoms):
        atom_type = np.random.choice(10, p=[0.45, 0.15, 0.2, 0.05, 0.03, 0.03, 0.02, 0.02, 0.01, 0.04])
        x_L[i, atom_type] = 1.0
        # Random continuous features for remaining dims
        x_L[i, 10:] = torch.randn(157) * 0.1
    
    # pos_L: 3D coordinates centered around origin (Angstroms)
    pos_L = torch.randn(num_atoms, 3) * 2.0  # ~2A spread
    
    # edge_index_L: bond connectivity (simple chain + random bonds)
    src, dst = [], []
    for i in range(num_atoms - 1):
        src.extend([i, i+1])
        dst.extend([i+1, i])
    # Add some random bonds for rings
    for _ in range(num_atoms // 4):
        a, b = np.random.choice(num_atoms, 2, replace=False)
        src.extend([int(a), int(b)])
        dst.extend([int(b), int(a)])
    edge_index_L = torch.tensor([src, dst], dtype=torch.long)
    
    return x_L, pos_L, edge_index_L

def generate_protein(num_residues=None):
    """Generate a synthetic protein pocket."""
    if num_residues is None:
        num_residues = np.random.randint(30, 120)  # Pocket size range
    
    # x_P: amino acid one-hot (21-dim: 20 AAs + unknown)
    x_P = torch.zeros(num_residues, 21)
    for i in range(num_residues):
        aa_type = np.random.randint(0, 20)
        x_P[i, aa_type] = 1.0
    
    # pos_P: C-alpha coordinates (pocket centered near origin)
    pos_P = torch.randn(num_residues, 3) * 8.0  # ~8A spread for pocket
    
    # normals_P: surface normals (unit vectors pointing outward)
    center = pos_P.mean(dim=0)
    normals_P = pos_P - center
    normals_P = normals_P / (normals_P.norm(dim=1, keepdim=True) + 1e-6)
    
    return x_P, pos_P, normals_P

def generate_sample():
    """Generate one protein-ligand complex as a PyG Data object."""
    x_L, pos_L, edge_index_L = generate_ligand()
    x_P, pos_P, normals_P = generate_protein()
    
    # Motif decomposition (simplified)
    num_motifs = max(1, x_L.size(0) // 5)
    atom_to_motif = torch.randint(0, num_motifs, (x_L.size(0),))
    joint_indices = torch.tensor(sorted(np.random.choice(
        x_L.size(0), min(num_motifs, x_L.size(0)), replace=False
    ).tolist()), dtype=torch.long)
    
    data = Data(
        x_L=x_L,
        pos_L=pos_L,
        edge_index_L=edge_index_L,
        x_P=x_P,
        pos_P=pos_P,
        normals_P=normals_P,
        pocket_center=pos_L.mean(dim=0, keepdim=True),
        atom_to_motif=atom_to_motif,
        joint_indices=joint_indices,
        num_motifs=torch.tensor([num_motifs], dtype=torch.long),
        pos_metals=None,
        num_nodes_L=x_L.size(0),
        num_nodes_P=x_P.size(0),
    )
    return data

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_samples", type=int, default=1000, help="Total samples")
    parser.add_argument("--shard_size", type=int, default=100, help="Samples per shard")
    args = parser.parse_args()
    
    os.makedirs(args.dir, exist_ok=True)
    
    num_shards = (args.num_samples + args.shard_size - 1) // args.shard_size
    manifest = []
    total = 0
    
    print(f"🔬 Generating {args.num_samples} synthetic protein-ligand complexes...")
    for shard_idx in range(num_shards):
        count = min(args.shard_size, args.num_samples - total)
        data_list = [generate_sample() for _ in range(count)]
        
        shard_file = f"shard_{shard_idx:04d}.pt"
        shard_path = os.path.join(args.dir, shard_file)
        torch.save(data_list, shard_path)
        
        manifest.append({
            "file": shard_file,
            "start_idx": total,
            "end_idx": total + count,
        })
        total += count
        print(f"  Shard {shard_idx+1}/{num_shards}: {count} samples saved.")
    
    manifest_path = os.path.join(args.dir, "shards_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    
    print(f"✅ Generated {total} samples across {num_shards} shards.")
    print(f"📄 Manifest saved to {manifest_path}")

if __name__ == "__main__":
    main()
