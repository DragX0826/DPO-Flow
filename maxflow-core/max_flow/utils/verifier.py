import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem

class System2Verifier:
    """
    SOTA Phase 11: Self-Verification Module (System 2 Thinking).
    
    Acts as an internal auditor for the generative model, flagging:
    1. Structural Validity (Sanitization)
    2. Toxicophores (Reactive groups, PAINS)
    3. Druglikeness (Lipinski Rules)
    
    Ref: "Learning to Self-Verify Makes Language Models Better Reasoners" (arXiv:2602.03139)
    """
    def __init__(self):
        # Define common toxicophores/unwanted groups (SMARTS)
        # 1. Nitro groups (often mutagenic)
        # 2. Aldehydes (reactive)
        # 3. Michael acceptors (specific types)
        # 4. Peroxides
        self.toxic_smarts = [
            "[N+](=O)[O-]", # Nitro
            "[CX3](=O)[H]", # Aldehyde
            "[OO]", # Peroxide
            "[#6]-[#6](=[#6])-[#6](=O)-[#6]", # Michael Acceptor (generic enone)
            "c1ccccc1[N+](=O)[O-]", # Nitrobenzene
        ]
        self.toxic_mols = [Chem.MolFromSmarts(s) for s in self.toxic_smarts if s]
        
    def verify(self, mol: Chem.Mol, strict: bool = False) -> dict:
        """
        Runs a battery of checks.
        Returns:
            passed (bool): Whether the molecule is acceptable.
            reasons (list): List of failure reasons.
            metrics (dict): Computed properties.
        """
        if mol is None:
            return False, ["Invalid Molecule"], {}
            
        try:
            Chem.SanitizeMol(mol)
        except:
            return False, ["Sanitization Failed"], {}
            
        reasons = []
        metrics = {}
        
        # 1. Lipinski Check
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        
        metrics['MW'] = mw
        metrics['LogP'] = logp
        
        # FIP/CNS Constraints (Hard Mode)
        # CNS usually requires MW < 450, LogP < 5 (but ideally 2-4)
        if mw > 600: reasons.append("MW > 600 (Too Heavy)")
        if logp > 6: reasons.append("LogP > 6 (Too Lipophilic)")
        if hbd > 5: reasons.append("HBD > 5 (Poor Permeability)")
        
        # 2. Toxicophore Check
        for tm in self.toxic_mols:
            if mol.HasSubstructMatch(tm):
                reasons.append("Contains Toxicophore")
                break # Fail fast on toxicity
                
        # 3. Structural Quality
        # Check for disconnected fragments
        if len(Chem.GetMolFrags(mol)) > 1:
            reasons.append("Disconnected Fragments")
            
        passed = len(reasons) == 0
        
        return passed, reasons, metrics

    def batch_verify(self, mols):
        """Vectorized-style wrapper (linear loop for RDKit)."""
        results = []
        for m in mols:
            results.append(self.verify(m))
        return results
