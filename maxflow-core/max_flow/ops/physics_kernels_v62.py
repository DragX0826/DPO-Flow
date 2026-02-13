# max_flow/ops/physics_kernels_v62.py

import torch
import triton
import triton.language as tl

@triton.jit
def fused_interaction_force_kernel(
    pL_ptr, pP_ptr, qL_ptr, qP_ptr, 
    forceL_ptr, # Output forces (N_L, 3)
    batchL_ptr, batchP_ptr,
    n_ligand, n_pocket,
    stride_pl, stride_pp,
    stride_fl,
    epsilon, sigma, dielectric,
    BLOCK_SIZE_L: tl.constexpr, BLOCK_SIZE_P: tl.constexpr,
):
    """
    Triton Kernel: Fused Energy & Analytical Force Calculation.
    Computes Lennard-Jones + Coulomb interaction and outputs forces directly.
    """
    # 1. Block indices
    pid_l = tl.program_id(0)
    
    offsets_l = pid_l * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    mask_l = offsets_l < n_ligand
    
    # Load ligand atoms (pos and charge)
    pl_x = tl.load(pL_ptr + offsets_l * 3 + 0, mask=mask_l)
    pl_y = tl.load(pL_ptr + offsets_l * 3 + 1, mask=mask_l)
    pl_z = tl.load(pL_ptr + offsets_l * 3 + 2, mask=mask_l)
    ql = tl.load(qL_ptr + offsets_l, mask=mask_l)
    bl = tl.load(batchL_ptr + offsets_l, mask=mask_l)
    
    # Accumulators for forces
    fx = tl.zeros([BLOCK_SIZE_L], dtype=tl.float32)
    fy = tl.zeros([BLOCK_SIZE_L], dtype=tl.float32)
    fz = tl.zeros([BLOCK_SIZE_L], dtype=tl.float32)

    # 2. Iterate over pocket atoms
    for start_p in range(0, n_pocket, BLOCK_SIZE_P):
        offsets_p = start_p + tl.arange(0, BLOCK_SIZE_P)
        mask_p = offsets_p < n_pocket
        
        pp_x = tl.load(pP_ptr + offsets_p * 3 + 0, mask=mask_p)
        pp_y = tl.load(pP_ptr + offsets_p * 3 + 1, mask=mask_p)
        pp_z = tl.load(pP_ptr + offsets_p * 3 + 2, mask=mask_p)
        qp = tl.load(qP_ptr + offsets_p, mask=mask_p)
        bp = tl.load(batchP_ptr + offsets_p, mask=mask_p)
        
        # Pairwise distance components
        # (BLOCK_L, BLOCK_P)
        dx = pl_x[:, None] - pp_x[None, :]
        dy = pl_y[:, None] - pp_y[None, :]
        dz = pl_z[:, None] - pp_z[None, :]
        
        dist_sq = dx*dx + dy*dy + dz*dz + 1e-6
        # Batch Mask
        mask_inter = (bl[:, None] == bp[None, :]) & mask_l[:, None] & mask_p[None, :]
        
        # 3. Analytical Forces
        # F = - dE/dr * r_vec / r
        # VdW (12-6): E = eps * [(sig/r)^12 - 2(sig/r)^6]
        # dE/dr = eps * [-12 sig^12 / r^13 + 12 sig^6 / r^7]
        #       = 12 * eps/r * [(sig/r)^6 - (sig/r)^12]
        
        inv_r2 = 1.0 / dist_sq
        inv_r = tl.sqrt(inv_r2)
        sig_r6 = (sigma * sigma * inv_r2) ** 3
        sig_r12 = sig_r6 * sig_r6
        
        # VdW Force component: -dE/dr_vdw
        f_vdw_scalar = (12.0 * epsilon * inv_r2) * (sig_r12 - sig_r6)
        
        # Coulomb: E = 332 * q1*q2 / (eps_d * r)
        # dE/dr = -332 * q1*q2 / (eps_d * r^2)
        # Force component: -dE/dr_coul = 332 * q1*q2 / (eps_d * r^3)
        f_coul_scalar = (332.06 * ql[:, None] * qp[None, :]) / (dielectric * dist_sq * inv_r)
        
        f_total_scalar = (f_vdw_scalar + f_coul_scalar) * mask_inter
        
        # Project force to components
        fx += tl.sum(f_total_scalar * dx, axis=1)
        fy += tl.sum(f_total_scalar * dy, axis=1)
        fz += tl.sum(f_total_scalar * dz, axis=1)

    # 4. Store forces
    tl.store(forceL_ptr + offsets_l * 3 + 0, fx, mask=mask_l)
    tl.store(forceL_ptr + offsets_l * 3 + 1, fy, mask=mask_l)
    tl.store(forceL_ptr + offsets_l * 3 + 2, fz, mask=mask_l)

def fused_physics_force_triton(pL, pP, qL, qP, batchL, batchP, epsilon=0.15, sigma=3.5, dielectric=80.0):
    """
    Python wrapper for the fused Triton kernel.
    """
    n_ligand = pL.shape[0]
    n_pocket = pP.shape[0]
    forceL = torch.zeros_like(pL)
    
    grid = (triton.cdiv(n_ligand, 128),)
    
    fused_interaction_force_kernel[grid](
        pL, pP, qL, qP,
        forceL,
        batchL, batchP,
        n_ligand, n_pocket,
        pL.stride(0), pP.stride(0),
        forceL.stride(0),
        epsilon, sigma, dielectric,
        BLOCK_SIZE_L=128, BLOCK_SIZE_P=128
    )
    
    return forceL
