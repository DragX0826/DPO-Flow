import os
from rdkit import Chem
from rdkit.Chem import AllChem

def create_mock_structures(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Mock PDB (Minimal)
    pdb_content = """ATOM      1  N   ALA A   1      -0.528   1.595   0.000  1.00  0.00           N  
ATOM      2  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C  
ATOM      3  C   ALA A   1       1.528   0.000   0.000  1.00  0.00           C  
ATOM      4  O   ALA A   1       2.152   1.074   0.000  1.00  0.00           O  
"""
    pdb_path = os.path.join(data_dir, "mock_pdb.pdb")
    with open(pdb_path, "w") as f:
        f.write(pdb_content)
        
    # 2. Mock SDF (Generate via RDKit for validity)
    # Create a simple Ethanol molecule as a mock ligand
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    
    sdf_path = os.path.join(data_dir, "mock_sdf.sdf")
    with open(sdf_path, "w") as f:
        # Use MolToMolBlock to get a valid V2000/V3000 string
        f.write(Chem.MolToMolBlock(mol))
    
    return pdb_path, sdf_path

if __name__ == "__main__":
    create_mock_structures("smoke_test_data")
    print("Mock structures created in smoke_test_data/")
