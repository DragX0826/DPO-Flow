import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import logging
from ..utils.esm import get_esm_model

logger = logging.getLogger("SAEB-Flow.core.model")

# --- 1. ENCODERS & PERCEPTION ---

class SinusoidalTimeEmbedding(nn.Module):
    """
    Multi-frequency sinusoidal time embedding.
    Maps scalar t in [0,1] to a rich feature vector, following DDPM/DiT conventions.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )
    
    def forward(self, t):
        """t: (B,) -> (B, dim)"""
        args = t.unsqueeze(-1) * self.freqs * 2 * math.pi
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class StructureSequenceEncoder(nn.Module):
    """
    Bridges ESM-2 sequence embeddings with 3D structural context.
    Expects x_P with first 4 dims = atom one-hots, remaining = ESM features.
    """
    def __init__(self, esm_model_name="esm2_t33_650M_UR50D", hidden_dim=64):
        super().__init__()
        self.esm_dim = 1280
        self.adapter = nn.Sequential(
            nn.Linear(self.esm_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        # Fallback for non-ESM features (e.g., one-hot only)
        self.fallback = nn.Sequential(
            nn.Linear(25, hidden_dim),
            nn.SiLU()
        )

    def forward(self, x_P):
        x_P = x_P.float()
        feat_dim = x_P.size(-1)
        if feat_dim > 25:
            # ESM features present: skip first 4 atom one-hots
            esm_slice = x_P[..., 4:4+self.esm_dim]
            if esm_slice.size(-1) < self.esm_dim:
                esm_slice = F.pad(esm_slice, (0, self.esm_dim - esm_slice.size(-1)))
            return self.adapter(esm_slice)
        else:
            # Fallback: use raw one-hot features
            if feat_dim < 25:
                x_P = F.pad(x_P, (0, 25 - feat_dim))
            return self.fallback(x_P[..., :25])


class GVPAdapter(nn.Module):
    """Cross-Attention for Protein-Ligand interaction with distance bias."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, x_L, x_P, dist_lp):
        """
        x_L: (B, N_L, D), x_P: (B, N_P, D), dist_lp: (B, N_L, N_P)
        """
        q = self.q_proj(x_L)
        k = self.k_proj(x_P)
        v = self.v_proj(x_P)
        
        # Distance-gated attention bias (mask out atoms > 10A)
        attn_bias = -1e9 * (dist_lp > 10.0).float()
        scores = torch.bmm(q, k.transpose(-1, -2)) / self.scale + attn_bias
        probs = F.softmax(scores, dim=-1)
        context = torch.bmm(probs, v)
        return x_L + context, probs


# --- 2. INNOVATIONS (CBSF) ---

class ShortcutFlowHead(nn.Module):
    """
    Confidence-Bootstrapped Shortcut Flow (CBSF) Head.
    Predicts tangent velocity, absolute endpoint (shortcut), and confidence.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.gn = nn.GroupNorm(min(8, hidden_dim), hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, 3, bias=False)
        self.x1_proj = nn.Linear(hidden_dim, 3)
        self.conf_proj = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, h, pos_L=None):
        """h: (B, N, D), pos_L: (B, N, 3)"""
        h_norm = self.gn(h.transpose(1, 2)).transpose(1, 2)  # GroupNorm over channels
        v_pred = self.v_proj(h_norm)
        x1_delta = self.x1_proj(h_norm)
        x1_pred = (pos_L + x1_delta) if pos_L is not None else x1_delta
        conf = self.conf_proj(h)
        return {'v_pred': v_pred, 'x1_pred': x1_pred, 'confidence': conf}


class RecyclingEncoder(nn.Module):
    """AF2-style recycling: encodes pairwise distances of previous pose."""
    def __init__(self, hidden_dim, num_rbf=16):
        super().__init__()
        self.dist_embed = nn.Linear(num_rbf, hidden_dim)
        centers = torch.linspace(0, 20, num_rbf)
        self.register_buffer("rbf_centers", centers)

    def forward(self, prev_pos_L, prev_latent):
        dist = torch.norm(prev_pos_L.unsqueeze(2) - prev_pos_L.unsqueeze(1), dim=-1)
        rbf = torch.exp(-0.5 * (dist.unsqueeze(-1) - self.rbf_centers).pow(2))
        h_recycling = self.dist_embed(rbf.mean(dim=2))
        return prev_latent + h_recycling


# --- 3. BACKBONE ---

class PermutationInvariantBlock(nn.Module):
    """Self-Attention + FFN block for unordered atom sets."""
    def __init__(self, d_model, nhead=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x):
        h, _ = self.attn(x, x, x)
        x = self.norm1(x + h)
        h = self.ff(x)
        return self.norm2(x + h)


class SAEBFlowBackbone(nn.Module):
    """
    Master Architecture: Perception -> Cross-Attention -> Self-Attention -> Policy Head.
    
    All outputs are shaped (B, N, *) — no flattening.
    """
    def __init__(self, node_in_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Linear(node_in_dim, hidden_dim)
        self.time_embed = SinusoidalTimeEmbedding(hidden_dim)
        self.perception = StructureSequenceEncoder(hidden_dim=hidden_dim)
        self.cross_attn = GVPAdapter(hidden_dim)
        self.recycling = RecyclingEncoder(hidden_dim)
        self.reasoning = nn.ModuleList([
            PermutationInvariantBlock(hidden_dim) for _ in range(num_layers)
        ])
        self.head = ShortcutFlowHead(hidden_dim)

    def forward(self, x_L, x_P, pos_L, pos_P, t, prev_pos_L=None, prev_latent=None):
        """
        Args:
            x_L: (B, N_L, D_L) ligand features
            x_P: (B, N_P, D_P) protein features  
            pos_L: (B, N_L, 3) ligand positions
            pos_P: (B, N_P, 3) protein positions
            t: (B,) time in [0, 1]
            prev_pos_L: optional (B, N_L, 3) for recycling
            prev_latent: optional (B, N_L, D) for recycling
        Returns:
            dict with 'v_pred' (B,N,3), 'x1_pred' (B,N,3), 'confidence' (B,N,1), 'latent' (B,N,D)
        """
        B, N, _ = pos_L.shape
        
        # 1. Embed ligand features + time conditioning
        h = self.embedding(x_L)                    # (B, N, D)
        t_emb = self.time_embed(t)                 # (B, D)
        h = h + t_emb.unsqueeze(1)                 # broadcast to all atoms
        
        # 2. Recycling injection (if available)
        if prev_pos_L is not None and prev_latent is not None:
            h = self.recycling(prev_pos_L, h)
        
        # 3. Protein perception + cross-attention
        h_P = self.perception(x_P)                 # (B, N_P, D)
        dist_lp = torch.norm(pos_L.unsqueeze(2) - pos_P.unsqueeze(1), dim=-1)
        h, _ = self.cross_attn(h, h_P, dist_lp)
        
        # 4. Self-attention reasoning (permutation invariant)
        for layer in self.reasoning:
            h = layer(h)
        
        # 5. Policy heads — output is (B, N, *)
        out = self.head(h, pos_L=pos_L)
        out['latent'] = h  # for recycling
        return out


class RectifiedFlow(nn.Module):
    """Rectified Flow (Liu et al., 2023) wrapper."""
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, **kwargs):
        return self.backbone(**kwargs)
