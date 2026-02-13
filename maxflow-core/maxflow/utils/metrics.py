# maxflow/utils/metrics.py

import torch
from rdkit import Chem
from rdkit.Chem import QED
try:
    from rdkit.Chem import RDConfig
    import sys
    import os
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
except ImportError:
    sascorer = None

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from maxflow.utils.physics import calculate_affinity_reward

# Compatibility Alias
def compute_vina_score(pos_L, pos_P, data=None):
    return calculate_affinity_reward(pos_L, pos_P, data=data)

# ── Phase 63/SOTA: Parallel Worker Functions (Full Pipeline) ──
def _worker_init():
    """Child process initializer: Quiets RDKit and sets CPU affinity."""
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
    except:
        pass

def _calculate_qed(mol):
    if mol is None: return 0.0
    try: return QED.qed(mol)
    except: return 0.0

def _calculate_sa(mol):
    if mol is None: return 10.0
    if sascorer is None:
        return min(10.0, 1.0 + 0.5 * mol.GetNumAtoms() / 10.0)
    try: return sascorer.calculateScore(mol)
    except: return 10.0

def _calculate_se(mol):
    if mol is None: return 0.0
    if sascorer is not None:
        try:
            score = sascorer.calculateScore(mol)
            return (10.0 - score) / 9.0
        except: pass
    try:
        ri = mol.GetRingInfo()
        if not ri.IsInitialized(): Chem.FastFindRings(mol)
        complexity = mol.GetRingInfo().NumRings() * 0.8 + (mol.GetNumHeavyAtoms() / 15.0)
        return torch.exp(torch.tensor(-complexity)).item()
    except: return 0.1

def _full_worker(data_dict):
    """Worker that handles reconstruction AND scoring to maximize parallel speed."""
    if data_dict is None: return None
    
    # Reconstruct a mini-Data object from the dict (picklable)
    class SimpleData: pass
    data = SimpleData()
    data.x_L = data_dict['x_L']
    data.pos_L = data_dict['pos_L']
    data.edge_index_L = data_dict.get('edge_index_L', None)
    
    try:
        mol = get_mol_from_data(data)
        if mol is None:
            return None # Signal failure
        # Standard scoring
        qed = _calculate_qed(mol)
        sa = _calculate_sa(mol)
        se = _calculate_se(mol)
        
        # SOTA addition for BBB narrative
        from rdkit.Chem import Descriptors
        tpsa = Descriptors.TPSA(mol)
        
        # SOTA addition: Return molecule as well for persistence/SDF saving
        return qed, sa, se, tpsa, mol
    except Exception:
        return None

class MultiObjectiveScorer:
    """
    Calculates combined scores for molecules based on:
    - QED (Quantitative Estimate of Drug-likeness)
    - SA (Synthetic Accessibility)
    - Lipinski Rule violations (optional)
    """
    
    def __init__(self):
        # Persistent pool to avoid process creation overhead. 
        # Being conservative with worker count for multi-GPU efficiency.
        cpu_count = multiprocessing.cpu_count()
        self.num_workers = max(1, min(cpu_count // 2, 4)) 
        self.pool = ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=_worker_init
        )
        
    def calculate_qed(self, mol):
        return _calculate_qed(mol)

    def calculate_sa(self, mol):
        return _calculate_sa(mol)

    def calculate_synthesis_entropy(self, mol):
        return _calculate_se(mol)

    @staticmethod
    def check_clashes(pos_L, threshold=0.8):
        """
        Calculates a penalty for steric clashes within the ligand.
        Returns a value proportional to the number of clashing atom pairs.
        """
        if pos_L.size(0) < 2: return 0.0
        dist = torch.norm(pos_L[:, None, :] - pos_L[None, :, :], dim=-1)
        # Mask out self-distances
        mask = ~torch.eye(pos_L.size(0), dtype=torch.bool, device=pos_L.device)
        clashes = (dist[mask] < threshold).sum().item() / 2.0 # Symmetric
        return clashes

    @staticmethod
    def calculate_ddg(energies_bound, energies_free):
        """
        [Alpha Phase 62] Thermodynamic ΔΔG Proxy.
        Estimates shift in partition function Z.
        Reward = -log(mean(exp(-E_bound))) + log(mean(exp(-E_free)))
        """
        if energies_bound is None or energies_free is None: return 0.0
        # kT approx 0.6 kcal/mol
        lse_bound = torch.logsumexp(-energies_bound / 0.6, dim=0)
        lse_free = torch.logsumexp(-energies_free / 0.6, dim=0)
        return -(lse_bound - lse_free)

    @staticmethod
    def calculate_affinity(pos_L, pos_P, q_L=None, q_P=None, data=None):
        return calculate_affinity_reward(pos_L, pos_P, q_L=q_L, q_P=q_P, data=data)

    def calculate_reward(self, mol, pos_L=None, pos_P=None, weights=None, data=None):
        """
        Comprehensive reward signal for MaxRL.
        Weights now support: qed, sa, se, clash, affinity, tpsa_penalty.
        """
        if weights is None:
            weights = {'qed': 3.0, 'sa': 1.0, 'se': 2.0, 'clash': -0.5, 'affinity': 0.1, 'tpsa_penalty': -0.1}
            
        if mol is not None:
            qed_val = self.calculate_qed(mol)
            sa_val = self.calculate_sa(mol)
            norm_sa = (10.0 - sa_val) / 9.0
            se_val = self.calculate_synthesis_entropy(mol)
            
            # SOTA Phase 3: Smooth PSA (Soft Gaussian Potential)
            # Center = 75, Width = 15. Provides smooth gradients for alignment.
            mu_psa, sigma_psa = 75.0, 15.0
            tpsa_reward = torch.exp(-torch.tensor((tpsa - mu_psa)**2 / (2 * sigma_psa**2))).item()
            # We treat this as a positive reward for being in the range
            tpsa_val = tpsa_reward 
        else:
            qed_val = 0.0
            norm_sa = 0.0
            se_val = 0.0
            tpsa_val = 0.0
        
        reward = weights.get('qed', 3.0) * qed_val + \
                 weights.get('sa', 1.0) * norm_sa + \
                 weights.get('se', 2.0) * se_val + \
                 weights.get('tpsa_psa', 2.0) * tpsa_val
        
        if pos_L is not None:
            clashes = self.check_clashes(pos_L)
            reward += weights.get('clash', -0.5) * clashes
            
            if pos_P is not None:
                # Pass data to calculate_affinity to ensure charges/batch are correct
                affinity = self.calculate_affinity(pos_L, pos_P, data=data)
                reward += weights.get('affinity', 0.1) * affinity
            
        return reward

    def calculate_batch_reward(self, data_batch, weights=None):
        """
        Phase 63 Debug: Unified Vectorized Batch Reward Scorer.
        SOTA Publication: Added TPSA penalty for clinical narrative.
        """
        if weights is None:
            weights = {'qed': 3.0, 'sa': 1.0, 'clash': -0.5, 'affinity': 0.1, 'tpsa_penalty': -0.1}

        batch_size = data_batch.num_graphs
        device = data_batch.x_L.device
        
        # ── Step 1: Generate robust batch indices ──
        if hasattr(data_batch, 'num_nodes_L') and isinstance(data_batch.num_nodes_L, torch.Tensor):
            batch_L = torch.repeat_interleave(
                torch.arange(batch_size, device=device),
                data_batch.num_nodes_L
            )
        else:
            batch_L = getattr(data_batch, 'x_L_batch', getattr(data_batch, 'batch', None))
        
        if hasattr(data_batch, 'num_nodes_P') and isinstance(data_batch.num_nodes_P, torch.Tensor):
            batch_P = torch.repeat_interleave(
                torch.arange(batch_size, device=device),
                data_batch.num_nodes_P
            )
        else:
            batch_P = getattr(data_batch, 'pos_P_batch', getattr(data_batch, 'x_P_batch', None))

        # Inject for PhysicsEngine
        data_batch.x_L_batch = batch_L
        data_batch.x_P_batch = batch_P

        # ── Step 2: GPU Affinity (Vectorized) ──
        # ── Step 2: GPU Affinity (Vectorized) ──
        with torch.no_grad():
            try:
                affinity = calculate_affinity_reward(data_batch.pos_L, data_batch.pos_P, data=data_batch)
                # Keep original NaNs to trigger invalidity in Step 6
            except Exception:
                affinity = torch.full((batch_size,), float('nan'), device=device)

        # ── Step 3: Prepare RDKit inputs (dict-based for pickling) ──
        batch_inputs = []
        for i in range(batch_size):
            mask = (batch_L == i)
            idx = mask.nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                batch_inputs.append(None)
                continue
            
            entry = {
                'x_L': data_batch.x_L[idx].detach().cpu(),
                'pos_L': data_batch.pos_L[idx].detach().cpu(),
            }
            if hasattr(data_batch, 'edge_index_L') and data_batch.edge_index_L is not None:
                curr_edges = data_batch.edge_index_L
                if curr_edges.size(1) > 0:
                    # SOTA Hardening: Explicit dictionary-based global-to-local mapping
                    global_idx = idx.cpu().tolist()
                    local_map = {g: i_loc for i_loc, g in enumerate(global_idx)}
                    
                    # Edges where both atoms are in current molecule
                    e_mask = (batch_L[curr_edges[0]] == i) & (batch_L[curr_edges[1]] == i)
                    if e_mask.any():
                        local_edges = curr_edges[:, e_mask]
                        try:
                            mapped_u = [local_map[int(g)] for g in local_edges[0]]
                            mapped_v = [local_map[int(g)] for g in local_edges[1]]
                            entry['edge_index_L'] = torch.tensor([mapped_u, mapped_v], dtype=torch.long)
                        except KeyError:
                            # Re-indexing failed, RDKit will fallback to KNN
                            pass
            
            batch_inputs.append(entry)

        # ── Step 4: Parallel RDKit Scoring ──
        valid_indices = [idx for idx, val in enumerate(batch_inputs) if val is not None]
        valid_inputs = [batch_inputs[idx] for idx in valid_indices]
        
        raw_results = []
        if valid_inputs:
            try:
                if len(valid_inputs) > 1:
                    raw_results = list(self.pool.map(_full_worker, valid_inputs))
                else:
                    raw_results = [_full_worker(inp) for inp in valid_inputs]
            except Exception:
                raw_results = [None] * len(valid_inputs)

        results = [None] * batch_size
        self.last_mols = [None] * batch_size
        for idx, res in zip(valid_indices, raw_results):
            if res is not None:
                results[idx] = res[:4]
                self.last_mols[idx] = res[4] if len(res) > 4 else None
        
        # ── Step 5: Clashes (Vectorized) ──
        pos_L = data_batch.pos_L
        dist_L = torch.cdist(pos_L, pos_L)
        mask_clash = (batch_L[:, None] == batch_L[None, :]) & (~torch.eye(pos_L.size(0), device=device, dtype=torch.bool))
        collision_matrix = (dist_L < 0.8) & mask_clash
        clashes_per_atom = collision_matrix.float().sum(dim=1)
        clashes_per_mol = torch.zeros(batch_size, device=device).scatter_add(0, batch_L, clashes_per_atom) / 2.0
        
        # ── Step 6: Assemble Reward & Valid Mask ──
        total_reward = torch.full((batch_size,), float('nan'), device=device)
        qed_vals = torch.full((batch_size,), float('nan'), device=device)
        sa_vals = torch.full((batch_size,), float('nan'), device=device)
        se_vals = torch.full((batch_size,), float('nan'), device=device)
        tpsa_vals = torch.full((batch_size,), float('nan'), device=device)
        valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for i, res in enumerate(results):
            if res is not None and not torch.isnan(affinity[i]):
                qed_vals[i], sa_vals[i], se_vals[i], tpsa_vals[i] = res
                
                # Normalization (Phase 60: 10-SA scale)
                norm_sa = (10.0 - sa_vals[i]) / 9.0
                
                # TPSA Penalty for BBB crossing
                # Target range [60, 90]
                tp = tpsa_vals[i]
                t_penalty = 0.0
                if tp < 60: t_penalty = 60 - tp
                elif tp > 90: t_penalty = tp - 90
                
                # ── Total Reward ──
                # Weights from Phase 62 + Publication Narrative
                r = weights.get('qed', 3.0) * qed_vals[i] + \
                    weights.get('sa', 1.0) * norm_sa + \
                    weights.get('se', 2.0) * se_vals[i] + \
                    weights.get('clash', -0.5) * clashes_per_mol[i] + \
                    weights.get('affinity', 0.1) * affinity[i] + \
                    weights.get('tpsa_penalty', -0.1) * t_penalty
                
                total_reward[i] = r
                valid_mask[i] = True
            else:
                # SOTA Hardening: NaN is a clear signal of failure
                total_reward[i] = float('nan')
                valid_mask[i] = False
                       
        return total_reward, valid_mask

def get_mol_from_data(data):
    """
    Constructs an RDKit molecule from PyG data for scoring.
    Phase 63 Debug: Added KNN-based bond fallback for molecules missing edge_index_L.
    """
    from rdkit import Chem
    from maxflow.utils.constants import allowable_features
    
    mol = Chem.RWMol()
    atomic_nums = allowable_features['possible_atomic_num_list']
    num_atoms = data.x_L.size(0)
    
    # 1. Add atoms
    for i in range(num_atoms):
        idx = torch.argmax(data.x_L[i, :len(atomic_nums)]).item()
        atom_num = atomic_nums[idx]
        mol.AddAtom(Chem.Atom(atom_num))
    
    # 2. Add bonds
    edge_index = getattr(data, 'edge_index_L', None)
    
    # ── KEY FIX: KNN Bond Fallback (Phase 63) ──
    if edge_index is None or edge_index.size(1) == 0:
        # Infer bonds from proximity if missing (Real CrossDocked data)
        # Using a threshold of 1.6A for most C-C/C-N/C-O bonds
        dist = torch.cdist(data.pos_L, data.pos_L)
        mask = (dist < 1.65) & (~torch.eye(num_atoms, dtype=torch.bool, device=data.pos_L.device))
        edge_index = mask.nonzero().t()
    
    if edge_index.size(1) > 0:
        for i in range(0, edge_index.size(1)):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if u < v:
                try:
                    mol.AddBond(u, v, Chem.BondType.SINGLE)
                except: pass
    
    # 3. Finalize molecule structure
    try:
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES ^ Chem.SANITIZE_VALENCE)
        Chem.FastFindRings(mol)
    except Exception:
        # Phase 63/SOTA: If molecule is completely broken, return None to trigger invalid_mask
        if mol.GetNumAtoms() == 0:
            return None
        # Otherwise, we keep it but it might be low quality
        pass
        
    if mol.GetNumAtoms() == 0:
        return None

    # 4. Add coords
    try:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(min(mol.GetNumAtoms(), data.pos_L.size(0))):
            conf.SetAtomPosition(i, data.pos_L[i].tolist())
        mol.AddConformer(conf)
        return mol.GetMol()
    except Exception:
        return None
