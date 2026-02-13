# max_flow/models/layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import math

class HyperConnection(nn.Module):
    def __init__(self, dim, init_scale=1.0):
        super().__init__()
        self.dim = dim
        self.W_d = nn.Parameter(torch.eye(dim) * init_scale)
        self.W_w = nn.Parameter(torch.eye(dim) * init_scale)
    def forward(self, x, residual):
        return torch.matmul(x, self.W_d.T) + torch.matmul(residual, self.W_w.T)

class GVPCrossAttention(MessagePassing):
    def __init__(self, s_dim, v_dim, num_heads=4):
        super().__init__(aggr='add', flow='source_to_target')
        self.s_dim = s_dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.head_dim = s_dim // num_heads
        self.q_proj = nn.Linear(s_dim, s_dim)
        self.k_proj = nn.Linear(s_dim, s_dim)
        self.v_proj = nn.Linear(s_dim, s_dim)
        self.out_proj = nn.Linear(s_dim, s_dim)
    def forward(self, x_L, x_P, edge_index):
        return self.propagate(edge_index, x=(x_L, x_P))
    def message(self, x_i, x_j):
        q = self.q_proj(x_i).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x_j).view(-1, self.num_heads, self.head_dim)
        attn = (q * k).sum(dim=-1) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        v = self.v_proj(x_j).view(-1, self.num_heads, self.head_dim)
        return (attn.unsqueeze(-1) * v).view(-1, self.s_dim)

class CausalMolSSM(nn.Module):
    """
    SOTA: Bidirectional Mamba-3 with Trapezoidal Discretization.
    Matches maxflow_pretrained.pt signature (195 weight tensors).
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, bidirectional=True):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.bidirectional = bidirectional

        # Forward Parameters
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.d_inner + d_state * 4, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        A_real = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(self.d_inner, 1) * -0.5
        A_imag = torch.pi * torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.complex(A_real, A_imag))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        if bidirectional:
            self.bwd_in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
            self.bwd_conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1)
            self.bwd_x_proj = nn.Linear(self.d_inner, self.d_inner + d_state * 4, bias=False)
            self.bwd_dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
            self.bwd_out_proj = nn.Linear(self.d_inner, d_model, bias=False)
            self.fusion = nn.Linear(d_model * 2, d_model)

    def forward(self, x, batch_idx=None):
        out_fwd = self._compute_ssm(x, batch_idx, direction='fwd')
        if not self.bidirectional: return out_fwd
        
        x_bwd = x.flip(dims=[0]) 
        out_bwd = self._compute_ssm(x_bwd, batch_idx.flip(dims=[0]) if batch_idx is not None else None, direction='bwd').flip(dims=[0])
        return self.fusion(torch.cat([out_fwd, out_bwd], dim=-1))

    def _compute_ssm(self, x, batch_idx, direction='fwd'):
        in_p, x_p, dt_p, conv, out_p = (self.in_proj, self.x_proj, self.dt_proj, self.conv1d, self.out_proj) if direction == 'fwd' else \
                                       (self.bwd_in_proj, self.bwd_x_proj, self.bwd_dt_proj, self.bwd_conv1d, self.bwd_out_proj)
        
        xz = in_p(x)
        x_ssm, z = xz.chunk(2, dim=-1)
        x_ssm = x_ssm.transpose(0, 1).unsqueeze(0)
        x_ssm = conv(x_ssm)[:, :, :x.size(0)].squeeze(0).transpose(0, 1)
        x_ssm = F.silu(x_ssm)

        ssm_params = x_p(x_ssm)
        delta, B_re, B_im, C_re, C_im = ssm_params.split([self.d_inner, self.d_state, self.d_state, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(dt_p(delta))
        y = self._scan_complex(x_ssm, delta, torch.complex(B_re, B_im), torch.complex(C_re, C_im))
        return out_p(y * F.silu(z))

    def _scan_complex(self, u, dt, B, C):
        A = -torch.exp(self.A_log)
        dt_c = dt.unsqueeze(-1).to(torch.complex64)
        A_c = A.unsqueeze(0)
        denom, numer = 2.0 - dt_c * A_c, 2.0 + dt_c * A_c
        log_A_bar = torch.log(numer + 1e-9) - torch.log(denom + 1e-9)
        u_bar = (2.0 * dt_c / (denom + 1e-9)) * B.unsqueeze(1) * u.unsqueeze(-1).to(torch.complex64)
        log_A_cumsum = torch.cumsum(log_A_bar, dim=0)
        decay = torch.exp(log_A_cumsum)
        decay_inv = torch.exp(-log_A_cumsum)
        H = (torch.cumsum(decay_inv * u_bar, dim=0)) * decay
        return (C.unsqueeze(1) * H).sum(dim=-1).real + self.D * u

SimpleS6 = CausalMolSSM
