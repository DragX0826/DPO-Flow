# max_flow/utils/visualization.py

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser, PDBIO
import os

def export_ligand_to_sdf(pos, atomic_numbers, save_path):
    """
    pos: (N, 3) tensor
    atomic_numbers: (N,) tensor or list
    """
    mol = Chem.RWMol()
    for at_num in atomic_numbers:
        mol.AddAtom(Chem.Atom(int(at_num)))
    
    conf = Chem.Conformer(len(atomic_numbers))
    for i in range(len(atomic_numbers)):
        conf.SetAtomPosition(i, pos[i].tolist())
    
    mol.AddConformer(conf)
    # Note: Bond inference is difficult without connectivity.
    # We can use RDKit's ConnectTheDots or assume the user will use PyMOL's 'connect'.
    
    with Chem.SDWriter(save_path) as writer:
        writer.write(mol)

def save_protein_with_ligand(protein_pdb, ligand_pos, ligand_atomic_numbers, output_dir, name="complex"):
    """
    Saves a combined visualization package.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Copy or Save Protein
    # (Assuming protein_pdb is already on disk)
    
    # 2. Save Ligand
    sdf_path = os.path.join(output_dir, f"{name}_ligand.sdf")
    export_ligand_to_sdf(ligand_pos, ligand_atomic_numbers, sdf_path)
    
    print(f"Visualization files saved to {output_dir}")
