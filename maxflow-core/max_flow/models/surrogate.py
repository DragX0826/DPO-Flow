# max_flow/models/surrogate.py
"""
[SOTA Publication Phase] Surrogate Scoring Module.
Enables High-Throughput Active Learning by replacing heavy physics/RDKit calls 
with a lightweight GNN proxy during the MaxRL alignment loop.
"""

import torch
import torch.nn as nn
from max_flow.models.backbone import CrossGVP

class GNNProxy(nn.Module):
    """
    Lightweight Surrogate Model for Reward Prediction.
    Trained to mimic the output of MultiObjectiveScorer.
    """
    def __init__(self, node_in_dim=58, hidden_dim=64):
        super().__init__()
        # Use a slim version of GVP for speed
        self.backbone = CrossGVP(
            node_in_dim=node_in_dim,
            hidden_dim=hidden_dim,
            num_layers=2 # Shallow for O(1) inference
        )
        
        # Output heads for various objectives
        self.head_affinity = nn.Sequential(nn.Linear(hidden_dim, 1))
        self.head_qed = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.head_sa = nn.Sequential(nn.Linear(hidden_dim, 1))
        self.head_tpsa = nn.Sequential(nn.Linear(hidden_dim, 1))

    def forward(self, data):
        """
        Predicts normalized scores for a batch of molecules.
        """
        # Encode geometry and chemistry
        h_node, _ = self.backbone(data)
        
        # Global Pooling (Mean)
        if hasattr(data, 'batch'):
            from torch_scatter import scatter_mean
            h_graph = scatter_mean(h_node, data.batch, dim=0)
        else:
            h_graph = h_node.mean(dim=0, keepdim=True)
            
        # Predict objectives
        affinity = self.head_affinity(h_graph).squeeze(-1)
        qed = self.head_qed(h_graph).squeeze(-1)
        sa = self.head_sa(h_graph).squeeze(-1)
        tpsa = self.head_tpsa(h_graph).squeeze(-1)
        
        return {
            'affinity': affinity,
            'qed': qed,
            'sa': sa,
            'tpsa': tpsa
        }

class EnsembleSurrogateScorer:
    """
    Phase 3: Uncertainty-Aware Reward Mechanism (UARM).
    Uses an ensemble of GNNs to predict rewards and their epistemic uncertainty.
    """
    def __init__(self, checkpoint_paths=None, device='cpu', num_models=3):
        self.device = device
        self.models = nn.ModuleList([GNNProxy().to(device) for _ in range(num_models)])
        if checkpoint_paths:
            for i, path in enumerate(checkpoint_paths):
                if i < len(self.models):
                    self.models[i].load_state_dict(torch.load(path, map_location=device))
        for m in self.models:
            m.eval()

    @torch.no_grad()
    def predict_batch_reward(self, data_batch, weights=None, lambda_u=0.5):
        """
        UARM Reward: R = average(R_i) - lambda_u * std(R_i)
        
        Args:
            data_batch: PyG Batch
            weights: Reward weights
            lambda_u: Uncertainty penalty weight (SOTA Robustness)
        """
        if weights is None:
            weights = {'qed': 3.0, 'sa': 1.0, 'affinity': 0.1, 'tpsa_penalty': -0.1}
            
        data_batch = data_batch.to(self.device)
        ensemble_rewards = []
        
        for model in self.models:
            preds = model(data_batch)
            norm_sa = (10.0 - preds['sa']) / 9.0
            
            # SOTA Phase 3: Smooth PSA (Soft Gaussian Potential)
            mu_psa, sigma_psa = 75.0, 15.0
            tpsa_reward = torch.exp(-(preds['tpsa'] - mu_psa)**2 / (2 * sigma_psa**2))
            
            reward_i = weights.get('qed', 3.0) * preds['qed'] + \
                      weights.get('sa', 1.0) * norm_sa + \
                      weights.get('affinity', 0.1) * preds['affinity'] + \
                      weights.get('tpsa_psa', 2.0) * tpsa_reward
            ensemble_rewards.append(reward_i)
            
        ensemble_rewards = torch.stack(ensemble_rewards, dim=0) # (num_models, B)
        
        mu_r = ensemble_rewards.mean(dim=0)
        sigma_r = ensemble_rewards.std(dim=0)
        
        # SOTN: UARM - Reward Hacking Protection
        uarm_reward = mu_r - lambda_u * sigma_r
        
        return uarm_reward, torch.ones(uarm_reward.size(0), dtype=torch.bool, device=self.device)

# Legacy alias
SurrogateScorer = EnsembleSurrogateScorer
