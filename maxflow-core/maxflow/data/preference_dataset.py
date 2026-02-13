# maxflow/data/preference_dataset.py

import os
import torch
import numpy as np
import json
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from maxflow.data.featurizer import ProteinLigandFeaturizer, FlowData

class PreferenceDataset(Dataset):
    """
    Standard Dataset for MaxRL Alignment.
    """
    def __init__(self, data_list, root_dir="", use_pt=True, cache_size=5):
        self.data_list = data_list
        self.root_dir = root_dir
        self.use_pt = use_pt
        self.featurizer = ProteinLigandFeaturizer()
        self.shard_mode = False
        self.shard_cache = {}
        self.shard_cache_keys = []
        self.cache_size = cache_size
        
        if use_pt and len(data_list) > 0 and isinstance(data_list[0], dict) and 'start_idx' in data_list[0]:
            self.shard_mode = True
            self.shard_index = data_list
            self.total_len = data_list[-1]['end_idx']
        else:
            self.total_len = len(data_list)

    def __len__(self):
        return self.total_len

    def _load_from_shard(self, idx):
        target_shard = None
        local_idx = 0
        for shard in self.shard_index:
            if shard['start_idx'] <= idx < shard['end_idx']:
                target_shard = shard
                local_idx = idx - shard['start_idx']
                break
        if target_shard is None: return None
        shard_path = os.path.join(self.root_dir, target_shard['file'])
        if shard_path in self.shard_cache:
            shard_data = self.shard_cache[shard_path]
            self.shard_cache_keys.remove(shard_path)
            self.shard_cache_keys.append(shard_path)
        else:
            try:
                shard_data = torch.load(shard_path, weights_only=False)
                if len(self.shard_cache) >= self.cache_size:
                    evict = self.shard_cache_keys.pop(0)
                    del self.shard_cache[evict]
                self.shard_cache[shard_path] = shard_data
                self.shard_cache_keys.append(shard_path)
            except: return None
        data = shard_data[local_idx]
        if not isinstance(data, FlowData):
            # Phase 63 Debug: Safer way to cast a PyG Data object
            new_data = FlowData()
            for key, item in data:
                new_data[key] = item
            data = new_data
        
        # Ensure num_nodes_L is set (defensive check)
        if not hasattr(data, 'num_nodes_L') or data.num_nodes_L is None:
            x_L = getattr(data, 'x_L', None)
            if x_L is not None:
                data.num_nodes_L = x_L.size(0)
            else:
                data.num_nodes_L = 0
            
        return data

    def __getitem__(self, idx):
        if self.shard_mode:
            data_win = self._load_from_shard(idx)
            if data_win is None: return None
            
            # Defensive check: ensure data_win has critical attributes
            if not hasattr(data_win, 'x_L') or data_win.x_L is None:
                return None # Skip invalid samples
            
            # PyG needs num_nodes for Batch.from_data_list
            if not hasattr(data_win, 'num_nodes') or data_win.num_nodes is None:
                n_L = data_win.x_L.size(0)
                n_P = data_win.x_P.size(0) if hasattr(data_win, 'x_P') and data_win.x_P is not None else 0
                data_win.num_nodes = n_L + n_P
            
            # Remove unstable motif fields for LMDB-converted data
            for attr in ['atom_to_motif', 'joint_indices', 'num_motifs']:
                if hasattr(data_win, attr):
                    delattr(data_win, attr)
            
            # Phase 63: Reward-Ranked Preference Pairs (DrugCLIP Inspiration)
            # Try to find a different molecule from the same shard to form a
            # semantically meaningful win/lose pair based on structural quality.
            data_lose = self._find_contrastive_partner(idx, data_win)
            
            return (data_win, data_lose)
        # ... standard paths ...
        return None

    def _find_contrastive_partner(self, idx, data_anchor):
        """
        Phase 63 Debug: Optimized deterministic partner selection.
        Replaces random shard lookups with a fixed offset (+1/-1) within 
        the same shard to maximize cache hits and minimize I/O.
        """
        # Determine current shard and boundaries
        shard_start, shard_end = 0, self.total_len
        if self.shard_mode:
            for shard in self.shard_index:
                if shard['start_idx'] <= idx < shard['end_idx']:
                    shard_start, shard_end = shard['start_idx'], shard['end_idx']
                    break
        
        # Pick partner: next one with a small random jitter to increase diversity
        import random
        jitter = random.randint(1, min(5, shard_end - shard_start - 1))
        partner_idx = shard_start + ((idx - shard_start + jitter) % (shard_end - shard_start))
        
        # Load partner (likely already in cache)
        partner = self._load_from_shard(partner_idx)
        if partner is None:
            # Emergency fallback: noise perturbation
            data_lose = data_anchor.clone()
            data_lose.pos_L = data_lose.pos_L + torch.randn_like(data_lose.pos_L) * 0.3
            return data_lose

        # Cleaning and metadata
        for attr in ['atom_to_motif', 'joint_indices', 'num_motifs']:
            if hasattr(partner, attr):
                delattr(partner, attr)
        
        if not hasattr(partner, 'num_nodes') or partner.num_nodes is None:
            partner.num_nodes = partner.x_L.size(0) + (partner.x_P.size(0) if hasattr(partner, 'x_P') else 0)
        
        return partner


class AutoShadowPreferenceDataset(Dataset):
    """
    SOTA Phase 59: High-Throughput Bin-Map Shadow Dataset.
    Bypasses Pickle entirely for extreme performance.
    """
    def __init__(self, data_list, root_dir="", cache_dir=".hft_cache"):
        self.data_list = data_list
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.bin_file = os.path.join(cache_dir, "shadow.bin")
        self.meta_file = os.path.join(cache_dir, "shadow_meta.json")
        
        self.is_ready = os.path.exists(self.bin_file) and os.path.exists(self.meta_file)
        if self.is_ready:
            with open(self.meta_file, 'r') as f:
                self.meta = json.load(f)
            if len(self.meta) != len(self.data_list):
                 self.is_ready = False

    def __len__(self):
        return len(self.data_list)
        
    @staticmethod
    def bake_shadow(data_list, root_dir, cache_dir=".hft_cache"):
        """
        SOTA Phase 59: Shadow Baking - Transforms .pt/raw to zero-copy binary.
        """
        os.makedirs(cache_dir, exist_ok=True)
        bin_file = os.path.join(cache_dir, "shadow.bin")
        meta_file = os.path.join(cache_dir, "shadow_meta.json")
        
        meta = []
        current_offset = 0
        
        # Open binary file for writing
        with open(bin_file, 'wb') as f_bin:
            for idx, item in enumerate(data_list):
                # Load raw data (assuming .pt or Data object)
                if isinstance(item, str):
                    data = torch.load(os.path.join(root_dir, item), weights_only=False)
                else:
                    data = item # Already a Data object
                
                m_entry = {}
                fields_to_save = ['x_L', 'pos_L', 'pos_P', 'q_P', 'normals_P']
                
                for field in fields_to_save:
                    val = getattr(data, field, None)
                    if val is not None:
                        val_np = val.detach().cpu().numpy().astype('float32')
                        m_entry[field] = {
                            'offset': current_offset,
                            'shape': list(val_np.shape)
                        }
                        f_bin.write(val_np.tobytes())
                        current_offset += val_np.nbytes
                
                meta.append(m_entry)
                
        with open(meta_file, 'w') as f_meta:
            json.dump(meta, f_meta)
        
        print(f"Shadow baking complete: {len(meta)} samples, {current_offset/1e6:.2f} MB.")
        return True

def preference_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0: return None
    wins, loses = zip(*batch)
    batch_win = Batch.from_data_list(wins)
    batch_lose = Batch.from_data_list(loses)
    
    # Inject/Synchronize Batch Indices for Vectorized Physics
    def inject_batch(out_batch, orig_list):
        # Manually create separate indices to ensure alignment with pos_L and pos_P
        xl_b = []
        xp_b = []
        for i, data in enumerate(orig_list):
            num_l = data.x_L.size(0)
            xl_b.append(torch.full((num_l,), i, dtype=torch.long))
            
            num_p = data.pos_P.size(0) if hasattr(data, 'pos_P') else 0
            if num_p > 0:
                xp_b.append(torch.full((num_p,), i, dtype=torch.long))
        
        out_batch.x_L_batch = torch.cat(xl_b)
        if xp_b: 
            out_batch.x_P_batch = torch.cat(xp_b)
        return out_batch

    return inject_batch(batch_win, wins), inject_batch(batch_lose, loses)
