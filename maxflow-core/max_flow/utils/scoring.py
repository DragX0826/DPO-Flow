# max_flow/utils/scoring.py

from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from rdkit.Contrib.SA_Score import sascorer
import torch

def get_molecular_metrics(sdf_path=None, mol=None):
    """
    Computes drug-likeness metrics for a ligand.
    """
    if sdf_path:
        suppl = Chem.SDMolSupplier(sdf_path)
        mol = next(suppl) if suppl else None
    
    if mol is None:
        return {"qed": 0.5, "sa": 5.0, "mw": 300.0} # Defaults
        
    try:
        qed_val = QED.qed(mol)
        sa_val = sascorer.calculateScore(mol)
        mw_val = Descriptors.MolWt(mol)
        return {"qed": qed_val, "sa": sa_val, "mw": mw_val}
    except:
        return {"qed": 0.0, "sa": 10.0, "mw": 0.0}

class MultiObjectiveMaxRLLoss(torch.nn.Module):
    """
    MaxRL Loss with weighted chemical properties.
    """
    def __init__(self, beta=0.1, w_qed=1.0, w_sa=0.5, w_affinity=0.2):
        super().__init__()
        self.beta = beta
        self.w_qed = w_qed
        self.w_sa = w_sa
        self.w_affinity = w_affinity

    def forward(self, MaxRL_logits, win_metrics, lose_metrics):
        """
        MaxRL_logits: (B,) from MaxRL (Err_lose - Err_win)
        win_metrics: dict of lists [qed, sa, affinity, ...] for winners
        lose_metrics: dict of lists for losers
        """
        # Multi-objective preference
        
        pref_qed = torch.tensor(win_metrics['qed']).to(MaxRL_logits.device) - torch.tensor(lose_metrics['qed']).to(MaxRL_logits.device)
        pref_sa = torch.tensor(win_metrics['sa']).to(MaxRL_logits.device) - torch.tensor(lose_metrics['sa']).to(MaxRL_logits.device)
        
        # Affinity Preference (Reward = -Energy, so higher reward = better)
        pref_affinity = torch.tensor(win_metrics.get('affinity', [0.0]*len(win_metrics['qed']))).to(MaxRL_logits.device) - \
                        torch.tensor(lose_metrics.get('affinity', [0.0]*len(lose_metrics['qed']))).to(MaxRL_logits.device)
        
        # total_logits = beta * MaxRL_Signal + sum(weight * delta_metric)
        # SA: Lower is better, so we want (lose - win) to be positive, weights for SA is applied to (lose-win) or (win-lose) with sign adjustment.
        # In this implementation: pref_sa = win - lose. If lose is higher (worse), pref_sa is negative.
        # We want to reward LOWER SA, so we should SUBTRACT pref_sa if it's positive.
        
        total_logits = MaxRL_logits + self.w_qed * pref_qed - self.w_sa * pref_sa + self.w_affinity * pref_affinity
        
        loss = -torch.nn.functional.logsigmoid(total_logits).mean()
        return loss
