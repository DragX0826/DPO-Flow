# maxflow/scripts/autonomous_oncology.py

import torch
import torch.nn.functional as F
from maxflow.models.max_rl import MaxRL, MaxFlow
from maxflow.utils.physics import PhysicsEngine
from torch_geometric.data import Batch
import time

class AutonomousOncologyExplorer:
    """
    SOTA Phase 36: Autonomous Online-RL discovery for cancer targets.
    Reduces the 'Druggability' gap by active exploration of the physics manifold.
    """
    def __init__(self, target_name="KRAS_G12D", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.target_name = target_name
        
        # Initialize Policy and Reference (Ref is frozen)
        self.model = MaxFlow().to(self.device)
        self.ref_model = MaxFlow().to(self.device)
        self.ref_model.load_state_dict(self.model.state_dict())
        self.ref_model.eval()
        
        self.MaxRL = MaxRL(self.model.flow, self.ref_model.flow, beta=0.1, clip_val=10.0, lambda_anchor=0.5)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        self.physics = PhysicsEngine()
        
        # Reward Normalization Stats (Phase 37)
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.alpha = 0.1 # Exponential moving average coefficient

    def run_discovery_cycle(self, iterations=10, n_samples=32):
        print(f"🎗️ Starting Autonomous Oncology Discovery for {self.target_name}...")
        
        for i in range(iterations):
            start_time = time.time()
            
            # 1. Create Mock Target Data (Oncology Context)
            data = self._get_oncology_target_data()
            batch = Batch.from_data_list([data for _ in range(n_samples)]).to(self.device)
            # Add batch indices
            batch.x_L_batch = torch.arange(n_samples, device=self.device).repeat_interleave(data.pos_L.size(0))
            batch.x_P_batch = torch.zeros(n_samples * data.pos_P.size(0), dtype=torch.long, device=self.device)
            
            # 2. Sample Candidates (Exploration)
            # Use high gamma for guidance-driven exploration
            self.model.eval()
            with torch.no_grad():
                sampled_pos, _ = self.model.flow.sample(batch, steps=20, gamma=2.0)
            
            # 3. Score candidates with Multi-Objective Physics
            rewards = self._score_candidates(sampled_pos, batch)
            
            # 4. Reward Normalization & Selection (Phase 37)
            # Update Running Stats
            curr_mean = rewards.mean().item()
            curr_std = rewards.std().item() + 1e-6
            self.reward_mean = (1 - self.alpha) * self.reward_mean + self.alpha * curr_mean
            self.reward_std = (1 - self.alpha) * self.reward_std + self.alpha * curr_std
            
            # Z-score Normalization
            norm_rewards = (rewards - self.reward_mean) / self.reward_std
            
            # Filter Winners and Losers for MaxRL
            sorted_idx = torch.argsort(norm_rewards, descending=True)
            n_pairs = n_samples // 4
            win_idx = sorted_idx[:n_pairs]
            lose_idx = sorted_idx[-n_pairs:]
            
            # 5. Policy Update (Stable Autonomous Learning)
            self.model.train()
            self.optimizer.zero_grad()
            
            # Use normalized rewards for MaxRL loss calculation
            # reward_win and reward_lose are passed to self.MaxRL.loss
            r_win = norm_rewards[win_idx]
            r_lose = norm_rewards[lose_idx]
            
            # Stable MaxRL Loss with anchored manifold
            loss = self.MaxRL.loss(batch, batch, reward_win=norm_rewards, reward_lose=norm_rewards * 0.5)
            
            loss.backward()
            self.optimizer.step()
            
            cycle_time = time.time() - start_time
            print(f"  Cycle {i+1}/{iterations} | Loss: {loss.item():.4f} | Max Reward: {rewards.max().item():.2f} | Time: {cycle_time:.1f}s")

    def _get_oncology_target_data(self):
        from torch_geometric.data import Data
        # Mocking KRAS G12D pocket
        num_atoms = 15
        num_res = 50
        data = Data(
            x_L=torch.randn(num_atoms, 167),
            pos_L=torch.randn(num_atoms, 3),
            x_P=torch.randn(num_res, 21),
            pos_P=torch.randn(num_res, 3),
            pocket_center=torch.zeros(1, 3),
            atom_to_motif=torch.zeros(num_atoms, dtype=torch.long)
        )
        if self.target_name == "KRAS_SELECTIVE":
            # Add Mutant and Wild-type references
            data.pos_P_mutant = data.pos_P.clone()
            data.pos_P_wildtype = data.pos_P.clone() + 0.5 
            
        # SOTA Phase 38: Covalent Warhead (e.g. Cys12)
        # Mocking a covalent pair: Ligand atom 0 -> Protein residue 12
        data.covalent_indices = torch.tensor([[0, 12]], dtype=torch.long)
            
        return data

    def _score_candidates(self, sampled_pos, batch):
        n_samples = batch.num_graphs
        n_atoms = sampled_pos.size(0) // n_samples
        rewards = []
        
        for k in range(n_samples):
            pos_k = sampled_pos[k*n_atoms : (k+1)*n_atoms]
            # 1. Base Physics Energy
            energy = self.physics.calculate_interaction_energy(pos_k, batch.pos_P[0:50]) # Mocked P
            
            # 2. Oncology Selectivity Reward (if applicable)
            selectivity = 0.0
            if self.target_name == "KRAS_SELECTIVE":
                 penalty = self.physics.calculate_selectivity_penalty(pos_k, batch.pos_P[0:50], batch.pos_P[0:50] + 0.5)
                 selectivity = -penalty.item()
            
            # 3. Covalent & ADMET Rewards (Phase 38)
            e_cov = self.physics.calculate_covalent_potential(pos_k, batch.pos_P[0:50], batch.covalent_indices)
            e_admet = self.physics.calculate_admet_scores(pos_k, batch.x_L[k*n_atoms : (k+1)*n_atoms])
            
            # Total Reward = -Energy + Selectivity - CovalentPenalty - ADMETPenalty
            rewards.append(-energy.item() + selectivity - e_cov.item() - e_admet.item())
            
        return torch.tensor(rewards, device=self.device)

if __name__ == "__main__":
    explorer = AutonomousOncologyExplorer(target_name="KRAS_SELECTIVE")
    explorer.run_discovery_cycle(iterations=5, n_samples=8)
