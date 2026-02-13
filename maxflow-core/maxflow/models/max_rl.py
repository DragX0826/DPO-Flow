# maxflow/models/max_rl.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxRL(nn.Module):
    """
    MaxFlow Engine: Maximum Likelihood RL for Flow Alignment.
    Replaces standard MaxRL with reward-weighted likelihood maximization.
    
    Key upgrades:
    1. InfoNCE contrastive loss for latent molecule-pocket alignment
    2. MaxRL objective for sparse reward exploration
    3. Decomposed loss return for transparent monitoring
    """
    def __init__(self, policy_model, ref_model, config=None, beta=0.1):
        super().__init__()
        self.policy = policy_model # RectifiedFlow object
        self.ref = ref_model       # RectifiedFlow object (frozen)
        
        # Use Config if provided, else fallback to args
        self.config = config
        self.beta = config.beta if config else beta
        self.lambda_geom = config.lambda_geom if config else 0.01
        self.clip_val = config.clip_val if config else 10.0
        self.lambda_anchor = config.lambda_anchor if config else 0.05
        self.lambda_clip = getattr(config, 'lambda_clip', 0.1) if config else 0.1
        self.lambda_atom = getattr(config, 'lambda_atom', 0.5) if config else 0.5 # SOTA Phase 65
        self.clip_temperature = getattr(config, 'clip_temperature', 0.07) if config else 0.07

    def _compute_infonce(self, z_L, z_P):
        """
        Phase 63: DrugCLIP-style InfoNCE Contrastive Loss.
        
        Forces the ligand latent of molecule i to be close to the pocket
        latent of pocket i (positive pair), and far from all other pockets
        in the batch (negative pairs). Provides B^2 gradient signals per batch.
        
        Args:
            z_L: (B, D) pooled ligand embeddings
            z_P: (B, D) pooled protein pocket embeddings
        Returns:
            loss_clip: scalar contrastive loss
        """
        # L2 normalize for cosine similarity
        z_L = F.normalize(z_L, dim=-1)
        z_P = F.normalize(z_P, dim=-1)
        
        # Cosine similarity matrix: (B, B)
        sim = torch.matmul(z_L, z_P.T) / self.clip_temperature
        
        # Positive pairs are on the diagonal (molecule i ↔ pocket i)
        labels = torch.arange(sim.size(0), device=sim.device)
        
        # Symmetric InfoNCE: L2P + P2L
        loss_l2p = F.cross_entropy(sim, labels)
        loss_p2l = F.cross_entropy(sim.T, labels)
        
        return (loss_l2p + loss_p2l) / 2.0

    def loss(self, data_win, data_lose, reward_win=None, reward_lose=None, valid_mask=None):
        """
        Phase 63/SOTA: BC-MaxRL Loss with Masked Components and Advantage Normalization.
        Only valid_mask samples contribute to MaxRL gradient.
        """
        # 1. Sample t for consistency across win/lose
        batch_size = data_win.num_graphs
        t = torch.rand(batch_size, device=data_win.x_L.device)
        
        # 2. Setup Flow variables (consistent noise x_0)
        # SOTA Hardening: Shape-Safe Consistent Noise (Pair-wise SNR boost)
        # Molecules in a pair often have different atom counts (N_W != N_L).
        # We use a vectorized Noise Pool to ensure consistency for common nodes.
        
        def get_num_nodes(data):
            if hasattr(data, 'num_nodes_L'): return data.num_nodes_L
            batch_L = getattr(data, 'x_L_batch', None)
            if batch_L is None: return torch.tensor([data.pos_L.size(0)], device=data.pos_L.device)
            # Use native torch.scatter_add for compatibility
            ones = torch.ones(data.pos_L.size(0), device=data.pos_L.device)
            num_nodes = torch.zeros(batch_size, device=data.pos_L.device)
            num_nodes.scatter_add_(0, batch_L, ones)
            return num_nodes.long()

        num_nodes_win = get_num_nodes(data_win)
        num_nodes_lose = get_num_nodes(data_lose)
        max_n = max(num_nodes_win.max().item(), num_nodes_lose.max().item())
        
        # Vectorized Noise Pool: (Batch, Max_N, 3)
        # This ensures that for the first N atoms that might correspond between win/lose,
        # the noise is identical, boosting preference signal stability.
        noise_pool = torch.randn(batch_size, int(max_n), 3, device=data_win.x_L.device)
        
        def expand_noise(pool, num_nodes):
            # Create a per-node noise tensor by indexing the pool
            # pool: (B, Max_N, 3), num_nodes: (B,)
            mask = torch.arange(max_n, device=num_nodes.device) < num_nodes.view(-1, 1) # (B, Max_N)
            # Flatten pool and mask to select active nodes
            return pool[mask] # (Total_Nodes, 3)

        noise_win = expand_noise(noise_pool, num_nodes_win)
        noise_lose = expand_noise(noise_pool, num_nodes_lose)

        # Process each path
        def center_data(data):
            # Guard for batch_size Derive from num_graphs if possible
            b_size = data.num_graphs if hasattr(data, 'num_graphs') else batch_size
            batch_L = getattr(data, 'x_L_batch', None)
            
            from maxflow.utils.scatter import robust_scatter_mean
            center_raw = robust_scatter_mean(data.pos_L, batch_L, dim=0, dim_size=b_size)
            
            if batch_L is not None:
                 center_nodes = center_raw[batch_L]
            else:
                 center_nodes = center_raw.expand(data.pos_L.size(0), -1)

            x_1 = data.pos_L - center_nodes
            data_centered = data.clone()
            data_centered.pos_L = x_1
            # data_centered.x_L_batch = batch_L # Already set or cloned
            
            # Pocket Centering
            if hasattr(data, 'num_nodes_P') and isinstance(data.num_nodes_P, torch.Tensor):
                batch_P = torch.repeat_interleave(
                    torch.arange(b_size, device=data.pos_P.device),
                    data.num_nodes_P
                )
            else:
                batch_P = getattr(data, 'x_P_batch', getattr(data, 'pos_P_batch', None))

            if batch_P is not None:
                 data_centered.pos_P = data.pos_P - center_raw[batch_P]
                 # data_centered.x_P_batch = batch_P
            else:
                 data_centered.pos_P = data.pos_P - center_raw
            return data_centered

        # WINNER
        data_win_c = center_data(data_win)
        x_t_w, x_0_w, x_1_w, t_nodes_w, _ = self.policy.sample_x_t(data_win_c, t, noise=noise_win)
        # LOSER
        data_lose_c = center_data(data_lose)
        x_t_l, x_0_l, x_1_l, t_nodes_l, _ = self.policy.sample_x_t(data_lose_c, t, noise=noise_lose)

        # 3. Compute Alignment & Confidence (Policy vs Ref)
        def compute_alignment_err(model, data_c, t_scalar, x_t, x_0, x_1, return_latent=False):
            orig_pos = data_c.pos_L
            data_c.pos_L = x_t
            
            if return_latent:
                result, conf_pred, kl_div, latents = model.backbone(t_scalar, data_c, return_latent=True)
            else:
                result, conf_pred, kl_div = model.backbone(t_scalar, data_c)
                latents = None
            
            data_c.pos_L = orig_pos
            
            # Robust Dictionary Unpacking
            v_model = result.get('v_pred')
            if v_model is None:
                v_trans, v_rot = result.get('v_trans'), result.get('v_rot')
                v_pred = v_trans[data_c.atom_to_motif] if hasattr(data_c, 'atom_to_motif') else v_trans
            else:
                v_pred = v_model
                
            atom_logits = result.get('atom_logits')

            v_target = x_1 - x_0
            error_flow = torch.sum((v_pred - v_target)**2, dim=-1)
            
            # SOTA Phase 65: Discrete Atom Preference
            error_atom = torch.zeros_like(error_flow)
            if atom_logits is not None and hasattr(data_c, 'atom_types'):
                error_atom = F.cross_entropy(atom_logits, data_c.atom_types, reduction='none')
            
            return error_flow, error_atom, v_pred, conf_pred, kl_div, latents

        # Winner errors
        err_flow_w, err_atom_w, v_pi_w, conf_pi_w, kl_pi_w, latents_w = compute_alignment_err(
            self.policy, data_win_c, t, x_t_w, x_0_w, x_1_w, return_latent=True
        )
        with torch.no_grad():
            err_ref_fw, err_ref_aw, _, _, _ = compute_alignment_err(self.ref, data_win_c, t, x_t_w, x_0_w, x_1_w)
            
        # Loser errors
        err_flow_l, err_atom_l, v_pi_l, conf_pi_l, kl_pi_l, _ = compute_alignment_err(
            self.policy, data_lose_c, t, x_t_l, x_0_l, x_1_l
        )
        with torch.no_grad():
            err_ref_fl, err_ref_al, _, _, _ = compute_alignment_err(self.ref, data_lose_c, t, x_t_l, x_0_l, x_1_l)

        # Unified error (Flow + Atom)
        # SOTA: weight atom log-prob significantly for categorical alignment
        err_pi_w = err_flow_w + self.lambda_atom * err_atom_w
        err_ref_w = err_ref_fw + self.lambda_atom * err_ref_aw
        err_pi_l = err_flow_l + self.lambda_atom * err_atom_l
        err_ref_l = err_ref_fl + self.lambda_atom * err_ref_al

        # 4. Use Robust Scatter Mean
        from maxflow.utils.scatter import robust_scatter_mean
        
        err_pi_w_graph = robust_scatter_mean(err_pi_w, getattr(data_win_c, 'x_L_batch', None), dim=0, dim_size=batch_size)
        err_ref_w_graph = robust_scatter_mean(err_ref_w, getattr(data_win_c, 'x_L_batch', None), dim=0, dim_size=batch_size)
        err_pi_l_graph = robust_scatter_mean(err_pi_l, getattr(data_lose_c, 'x_L_batch', None), dim=0, dim_size=batch_size)
        err_ref_l_graph = robust_scatter_mean(err_ref_l, getattr(data_lose_c, 'x_L_batch', None), dim=0, dim_size=batch_size)
        
        # 5. Masked MaxRL Loss
        delta_win = err_pi_w_graph - err_ref_w_graph
        delta_lose = err_pi_l_graph - err_ref_l_graph
        
        logits = self.beta * (delta_lose - delta_win)
        logits = torch.clamp(logits, -self.clip_val, self.clip_val)
        
        if valid_mask is None:
            valid_mask = torch.ones_like(logits, dtype=torch.bool)

        if getattr(self.config, 'use_maxrl', False):
            # MaxFlow Phase 11 & 12: MaxRL is the core objective
            # Weight = Reward / Baseline
            # Weight = Reward / Baseline (or similar positive reweighting)
            
            # 1. Compute baseline (batch mean of wins)
            if reward_win is not None:
                baseline = reward_win.mean() + 1e-8
                # 2. Compute Advantage (Ratio)
                # We assume reward_win is positive (e.g. Vina score transformed or normalized)
                # If using standard Vina (negative is better), ensure reward input is already inverted/normalized.
                # Here we trust the reward_scaler output which is typically normalized/standardized.
                # To be safe for MaxRL which expects >0 weights, we use softmax or exp if raw scores are used,
                # but with scaler, we might get negative values.
                # Robust approach: weight = exp(reward_win - baseline) to ensure positivity and relative importance
                # OR if user specified "reward / baseline", we assume positive rewards.
                # Let's use a stable reweighting: weight = sigmoid((r - mean) * scale) + 0.5?
                # User said: "reward / baseline". This implies Rewards > 0.
                # Let's shift rewards to be positive if necessary or use softmax weights.
                # Softmax over batch is a good proxy for "reward / sum(rewards)".
                
                # Implementation: Normalized Reweighting
                # w_i = exp(r_i / T) / sum(exp(r_j / T)) * BatchSize
                # This keeps mean weight around 1.0 but emphasizes high rewards.
                weights = F.softmax(reward_win / 1.0, dim=0) * batch_size # T=1.0 default
                weights = weights.detach() # Gradient does not flow through weights in MaxRL
                
                # 3. Loss = Weighted NLL
                # err_pi_w is our "Energy" (NLL equivalent for Flow Matching)
                # Minimize (Weight * Energy)
                loss_MaxRL_all = weights * err_pi_w
                loss_MaxRL = loss_MaxRL_all[valid_mask].mean() if valid_mask.any() else torch.tensor(0.0, device=logits.device)
            else:
                # Fallback if no rewards provided (should not happen in MaxRL loop)
                loss_MaxRL = err_pi_w[valid_mask].mean()
                
        elif reward_win is not None and reward_lose is not None:
             reward_diff = (reward_win - reward_lose)
             # Apply Masked Loss
             loss_MaxRL_all = -(reward_diff * F.logsigmoid(logits))
             loss_MaxRL = loss_MaxRL_all[valid_mask].mean() if valid_mask.any() else torch.tensor(0.0, device=logits.device)
        else:
             loss_MaxRL = -F.logsigmoid(logits[valid_mask]).mean() if valid_mask.any() else torch.tensor(0.0, device=logits.device)

        # 5.1 Anchoring & Regularization
        loss_anchor = err_pi_w.mean()
        loss_geom = v_pi_w.pow(2).mean() + v_pi_l.pow(2).mean()
        
        # 7. Bond loss (disabled for batched training — same as before)
        loss_bond = torch.tensor(0.0, device=data_win.x_L.device)
        
        # 8. Phase 63: DrugCLIP-Inspired Contrastive Loss
        loss_clip = torch.tensor(0.0, device=data_win.x_L.device)
        if latents_w is not None and batch_size > 1:
            z_L, z_P = latents_w
            # Explicit shape check to avoid silent failure
            if z_L.shape[0] == z_P.shape[0] and z_L.shape[0] > 1:
                loss_clip = self._compute_infonce(z_L, z_P)
            else:
                torch.cuda.empty_cache() # Heuristic for tiny batches
        
        # 9. Confidence & VIB Loss
        loss_conf = torch.tensor(0.0, device=data_win.x_L.device)
        if reward_win is not None and reward_lose is not None:
             r_win_sat = torch.tanh(reward_win / 10.0) * 10.0
             r_lose_sat = torch.tanh(reward_lose / 10.0) * 10.0
             loss_conf = F.mse_loss(conf_pi_w, r_win_sat) + F.mse_loss(conf_pi_l, r_lose_sat)
             
        lambda_kl = 0.001
        lambda_conf = 0.1
        loss_vib = kl_pi_w.mean() + kl_pi_l.mean()
        
        # 10. Total Loss (Phase 63: Rebalanced)
        # SOTA Phase 4: Support for individual loss return for PCGrad/AdaptiveLoss
        individual_losses = {
            'MaxRL': loss_MaxRL,
            'geom': self.lambda_geom * (loss_geom + loss_bond),
            'conf': lambda_conf * loss_conf,
            'vib': lambda_kl * loss_vib,
            'anchor': self.lambda_anchor * loss_anchor,
            'clip': self.lambda_clip * loss_clip
        }
        
        loss_total = sum(individual_losses.values())
        
        # Phase 63: Decomposed loss dict for transparent monitoring
        loss_dict = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in individual_losses.items()}
        
        return loss_total, loss_dict, individual_losses

class MaxFlow(nn.Module):
    """
    MaxFlow: Universal Engine Wrapper for Inference and Distillation.
    Combined Rectified Flow + Mamba-3 Trinity.
    """
    def __init__(self, node_in_dim=58, hidden_dim=128, num_layers=4):
        super().__init__()
        from maxflow.models.backbone import CrossGVP
        from maxflow.models.flow_matching import RectifiedFlow
        
        self.backbone = CrossGVP(node_in_dim=node_in_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        self.flow = RectifiedFlow(self.backbone)

    def forward(self, t, data):
        return self.backbone(t, data)
