"""
Quality Assessment Utilities for Molecular Generation
Provides functions to evaluate the quality and consistency of generated molecules.
"""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem


def calculate_molecule_quality(molecule_tensor):
    """
    Calculate quality score for a generated molecule.
    
    Args:
        molecule_tensor: Tensor representing the molecule (shape: [atoms, features])
        
    Returns:
        quality_score: Score between 0 and 1 (higher is better)
    """
    # Convert tensor to RDKit molecule
    mol = tensor_to_rdkit_molecule(molecule_tensor)
    
    if mol is None:
        return 0.0
    
    # Calculate multiple quality metrics
    metrics = {
        'qed': calculate_qed(mol),
        'sa': calculate_sa(mol),
        'synthesizability': calculate_synthesizability(mol),
        'druglikeness': calculate_druglikeness(mol),
        'validity': 1.0 if is_valid_molecule(mol) else 0.0
    }
    
    # Weighted average of metrics
    weights = {
        'qed': 0.3,
        'sa': 0.2,
        'synthesizability': 0.2,
        'druglikeness': 0.2,
        'validity': 0.1
    }
    
    quality_score = sum(metrics[k] * weights[k] for k in metrics)
    return max(min(quality_score, 1.0), 0.0)


def tensor_to_rdkit_molecule(tensor):
    """
    Convert a molecule tensor to RDKit molecule.
    
    Args:
        tensor: Tensor of shape [atoms, features]
        
    Returns:
        mol: RDKit molecule or None if conversion fails
    """
    try:
        # Assuming features include atom types and coordinates
        # This is a simplified conversion - in practice, you'd need proper atom typing
        num_atoms = tensor.shape[0]
        
        # Create empty molecule
        mol = Chem.RWMol()
        
        # Add atoms (simplified - in practice, use proper atom typing)
        for i in range(num_atoms):
            atom_type = int(tensor[i, 0].item()) % 100  # Simplified atom type
            mol.AddAtom(Chem.Atom(atom_type))
        
        # Add bonds (simplified - in practice, use proper connectivity)
        # For now, create a chain structure
        for i in range(num_atoms - 1):
            mol.AddBond(i, i + 1, Chem.BondType.SINGLE)
        
        # Set coordinates
        conf = Chem.Conformer(num_atoms)
        for i in range(num_atoms):
            conf.SetAtomPosition(i, (
                tensor[i, 1].item(),
                tensor[i, 2].item(),
                tensor[i, 3].item()
            ))
        mol.AddConformer(conf)
        
        # Sanitize
        Chem.SanitizeMol(mol)
        
        return mol
        
    except Exception as e:
        return None


def calculate_qed(mol):
    """
    Calculate Quantitative Estimate of Druglikeness (QED).
    
    Args:
        mol: RDKit molecule
        
    Returns:
        qed_score: QED score between 0 and 1
    """
    try:
        return Descriptors.QED(mol)
    except:
        return 0.0


def calculate_sa(mol):
    """
    Calculate Synthetic Accessibility (SA) score.
    
    Args:
        mol: RDKit molecule
        
    Returns:
        sa_score: SA score (lower is better, 1-10 scale)
    """
    try:
        return 1.0 / (1.0 + AllChem.SynthAccess(mol))
    except:
        return 0.0


def calculate_synthesizability(mol):
    """
    Calculate synthesizability score.
    
    Args:
        mol: RDKit molecule
        
    Returns:
        synthesizability_score: Score between 0 and 1
    """
    try:
        # Simplified synthesizability check
        heavy_atoms = len([atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1])
        ring_count = Chem.rdMolDescriptors.CalcNumRings(mol)
        
        # Reasonable molecules have 5-50 heavy atoms and <5 rings
        if 5 <= heavy_atoms <= 50 and ring_count < 5:
            return 1.0
        return 0.5
        
    except:
        return 0.0


def calculate_druglikeness(mol):
    """
    Calculate druglikeness score.
    
    Args:
        mol: RDKit molecule
        
    Returns:
        druglikeness_score: Score between 0 and 1
    """
    try:
        # Check Lipinski's rule of 5
        mol_weight = Descriptors.MolWt(mol)
        hba = Descriptors.NumHAcceptors(mol)
        hbd = Descriptors.NumHDonors(mol)
        logp = Descriptors.MolLogP(mol)
        
        violations = 0
        if mol_weight > 500: violations += 1
        if hba > 10: violations += 1
        if hbd > 5: violations += 1
        if logp > 5: violations += 1
        
        # Score: 1.0 for 0 violations, decreasing with violations
        return max(0.0, 1.0 - 0.25 * violations)
        
    except:
        return 0.0


def is_valid_molecule(mol):
    """
    Check if a molecule is valid.
    
    Args:
        mol: RDKit molecule
        
    Returns:
        is_valid: Boolean indicating if molecule is valid
    """
    try:
        return mol is not None and mol.GetNumAtoms() > 0
    except:
        return False


def calculate_consistency(traj):
    """
    Calculate consistency score for a trajectory.
    Measures how straight the path is from x_0 to x_1.
    
    Args:
        traj: Trajectory tensor of shape (steps, features)
    
    Returns:
        consistency_score: Score between 0 and 1 (higher is better)
    """
    if traj.shape[0] < 2:
        return 0.0
    
    # Calculate the straight line distance
    straight_distance = torch.norm(traj[-1] - traj[0]).item()
    
    # Calculate the actual path length
    actual_distance = 0.0
    for i in range(1, traj.shape[0]):
        actual_distance += torch.norm(traj[i] - traj[i-1]).item()
    
    # Consistency score: ratio of straight to actual distance
    if actual_distance == 0:
        return 0.0
    
    consistency_ratio = straight_distance / actual_distance
    return min(max(consistency_ratio, 0.0), 1.0)


def calculate_uncertainty_ensemble(predictions):
    """
    Calculate uncertainty from an ensemble of predictions.
    
    Args:
        predictions: List of prediction tensors from different models
        
    Returns:
        uncertainty: Uncertainty score between 0 and 1
    """
    if len(predictions) < 2:
        return 0.0
    
    # Calculate variance across ensemble
    predictions = torch.stack(predictions)
    variance = torch.var(predictions, dim=0).mean()
    
    # Normalize to [0, 1]
    return torch.sigmoid(variance * 10).item()  # Scale factor 10
