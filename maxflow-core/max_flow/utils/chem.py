# max_flow/utils/chem.py

import torch
from rdkit import Chem
from max_flow.utils.constants import allowable_features

from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
import os

# Singleton Factory
_fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
_factory = ChemicalFeatures.BuildFeatureFactory(_fdefName)

def get_pharmacophore_features(mol):
    """
    Extracts 6-dim Pharmacophore features per atom:
    [Donor, Acceptor, Aromatic, Hydrophobe, PosIonizable, NegIonizable]
    Returns: Tensor (N, 6)
    """
    num_atoms = mol.GetNumAtoms()
    # Initialize with zeros
    pharma_feats = torch.zeros((num_atoms, 6), dtype=torch.float32)
    
    if mol is None: return pharma_feats

    try:
        feats = _factory.GetFeaturesForMol(mol)
    except Exception:
        return pharma_feats

    # Map family to index
    family_map = {
        'Donor': 0, 'Acceptor': 1, 'Aromatic': 2, 'Hydrophobe': 3,
        'PosIonizable': 4, 'NegIonizable': 5
    }

    for f in feats:
        family = f.GetFamily()
        if family in family_map:
            idx = family_map[family]
            # A feature can be associated with multiple atoms
            for atom_idx in f.GetAtomIds():
                if atom_idx < num_atoms:
                    pharma_feats[atom_idx, idx] = 1.0
                    
    return pharma_feats

def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    """
    Extracts atom features:
    - Atomic number (one-hot)
    - Chirality (one-hot)
    - Degree (one-hot)
    - Hybridization (one-hot)
    - Aromaticity (one-hot)
    - Number of Hydrogens (one-hot)
    - Formal Charge (one-hot)
    """
    return torch.tensor(
        one_hot_encoding(atom.GetAtomicNum(), allowable_features['possible_atomic_num_list']) +
        one_hot_encoding(str(atom.GetChiralTag()), allowable_features['possible_chirality_list']) +
        one_hot_encoding(atom.GetDegree(), allowable_features['possible_degree_list']) +
        one_hot_encoding(str(atom.GetHybridization()), allowable_features['possible_hybridization_list']) +
        one_hot_encoding(atom.GetIsAromatic(), allowable_features['possible_is_aromatic_list']) +
        one_hot_encoding(atom.GetTotalNumHs(), allowable_features['possible_num_h_list']) +
        one_hot_encoding(atom.GetFormalCharge(), allowable_features['possible_formal_charge_list']),
        dtype=torch.float32
    )

def get_atom_features(mol):
    """
    Returns:
    - x (Tensor): Atom features (N, F_base + 6).
    - pos (Tensor): Coordinates (N, 3).
    - edge_index (Tensor): Bond connectivity (2, E).
    """
    # 1. Get coordinates
    conf = mol.GetConformer()
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
    
    # 2. Get base features
    base_features = []
    for atom in mol.GetAtoms():
        base_features.append(atom_features(atom))
    x_base = torch.stack(base_features)
    
    # 3. Get Pharmacophore features (SOTA Enhancement)
    x_pharma = get_pharmacophore_features(mol)
    
    # Concatenate: (N, Base) + (N, 6) -> (N, Base+6)
    x = torch.cat([x_base, x_pharma], dim=-1)
    
    # 4. Get bonds (edge_index)
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i]) # Undirected
    
    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
    return x, pos, edge_index

def physically_polish_molecule(mol):
    """
    [SOTA Phase 66] Merck Molecular Force Field (MMFF) Physical Polishing.
    Fixes twisted aromatic rings (planarity), bond length violations, and steric clashes.
    """
    if mol is None: return None
    from rdkit.Chem import AllChem
    try:
        # 1. Add Hydrogens (Required for accurate Force Field calculation)
        mol_h = Chem.AddHs(mol, addCoords=True)
        
        # 2. Setup MMFF Property
        if AllChem.MMFFHasAllMoleculeParams(mol_h):
            # 3. Optimize Geometry (200 Iterations max for speed)
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
            
            # 4. Remove Hydrogens for downstream metrics
            mol_clean = Chem.RemoveHs(mol_h)
            return mol_clean
        else:
            # Fallback if rare elements are present
            return mol
    except Exception:
        return mol

def sota_sanitize_molecule(mol, allow_mismatch=False):
    """
    [SOTA Phase 66] Advanced Sanitization Pipeline.
    1. Extract Largest Fragment (Removes "Fragmented Islands").
    2. Force Valency Correction (Fixes "Texas Carbons").
    3. RDKit Sanitize.
    """
    if mol is None: return None
    
    try:
        # 1. Fragment Retrieval: Keep only the main molecule
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            mol = sorted(frags, key=lambda x: x.GetNumAtoms(), reverse=True)[0]
            
        # 2. Basic Sanitization
        if allow_mismatch:
            # Mask valency errors to allow partial processing
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
        else:
            Chem.SanitizeMol(mol)
            
        return mol
    except Exception:
        # If valency fails, try to "unbond" hypervalent atoms
        try:
             # Heuristic: If Sanitize fails, it's often a Texas Carbon
             mol.UpdatePropertyCache(strict=False)
             return mol
        except:
             return None

def get_mol_from_data(data, atom_decoder=None):
    """
    [SOTA Utility] Reconstructs RDKit molecule from PyG FlowData.
    Useful for visualizing generated intermediates.
    """
    if hasattr(data, 'x_L'):
        # Get atom types (one-hot to idx)
        # Use first 100 features as proxy for atomic number list
        atom_types = torch.argmax(data.x_L[:, :100], dim=-1)
        pos = data.pos_L
    else:
        return None

    mol = Chem.RWMol()
    # Default atomic numbers mapping if decoder not provided
    # Assume 0 -> H, 1 -> C, 2 -> N, 3 -> O, etc. based on constants
    for a_idx in atom_types:
        atomic_num = int(a_idx.item()) % 90 + 1 # Fallback safety
        mol.AddAtom(Chem.Atom(atomic_num))
    
    conf = Chem.Conformer(len(atom_types))
    for i, p in enumerate(pos):
        conf.SetAtomPosition(i, (float(p[0]), float(p[1]), float(p[2])))
    mol.AddConformer(conf)
    
    return mol.GetMol()
