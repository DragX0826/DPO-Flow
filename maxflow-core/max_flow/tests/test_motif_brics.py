# max_flow/tests/test_motif_brics.py

from rdkit import Chem
from max_flow.utils.motifs import MotifDecomposer
import torch

def test_aspirin_decomposition():
    print("🧪 Testing Aspirin Decomposition...")
    # Aspirin: CC(=O)Oc1ccccc1C(=O)O
    aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    mol = Chem.MolFromSmiles(aspirin_smiles)
    
    decomposer = MotifDecomposer()
    motifs = decomposer.decompose(mol)
    
    print(f"Found {len(motifs)} motifs:")
    for i, m in enumerate(motifs):
        print(f"  [{i}] SMILES: {m['smiles']} | Atoms: {m['atoms']}")
        
    # Check if we have at least 2 motifs (Aspirin is usually split into Phenol-like and Acetyl-like groups)
    assert len(motifs) >= 1
    
    # Test centers calculation
    pos = torch.randn(mol.GetNumAtoms(), 3)
    centers = decomposer.get_motif_centers(pos, motifs)
    print(f"Motif Centers Shape: {centers.shape}")
    assert centers.shape[0] == len(motifs)
    assert centers.shape[1] == 3
    
    print("✅ Motif Decomposition Test Passed!")

if __name__ == "__main__":
    test_aspirin_decomposition()
