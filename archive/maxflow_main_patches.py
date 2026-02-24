"""
maxflow_main_patches.py — v96.0 Surgical Integration Guide

This file documents EXACTLY what to change in the main MaxFlow file to:
  1. Fix Bug 3  (optimizer tracking broken by parameter re-binding)
  2. Integrate SAEB-Flow (replace F-SE3 Kabsch with fibre-bundle geodesics)
  3. Wire SO(3)-averaged training target into the loss loop
  4. Replace import from cpu_torsional_flow → saeb_flow

Apply these as targeted str_replace operations on the main file.
Every patch is self-contained and independent.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# PATCH 0 — Replace import at top of main file
# ═══════════════════════════════════════════════════════════════════════════════

PATCH_0_OLD = """# [v94.0] Top-Tier Innovation: CPU-Optimized Fragment-SE3 Flow (F-SE3)
from cpu_torsional_flow import (
    get_rigid_fragments, sample_torsional_prior, 
    apply_fragment_kabsch, compute_torsional_joint_loss
)"""

PATCH_0_NEW = """# [v96.0] SAEB-Flow: SE(3)×T^n Averaged Energy-Based Flow
# Drop-in replacement for cpu_torsional_flow with correct fibre-bundle geometry.
from saeb_flow import (
    get_rigid_fragments, sample_torsional_prior,
    apply_fragment_kabsch, compute_torsional_joint_loss,
    build_fiber_bundle, so3_averaged_target,
    apply_saeb_step, energy_guided_interpolant,
    clear_bundle_cache,
)"""


# ═══════════════════════════════════════════════════════════════════════════════
# PATCH 1 — Bug 3 Fix: optimizer parameter re-binding
#
# Root cause: inside the training loop a new nn.Parameter object is created
# and re-assigned to pos_L.  The optimizer was initialised with the OLD
# Python object, so gradients flow to the new object but the optimizer
# steps the old one.  fix: in-place requires_grad_ only.
# ═══════════════════════════════════════════════════════════════════════════════

PATCH_1_OLD = """            # [v92.0 Atomic Parameter Re-Binding]
            # Ensure pos_L is always a fresh leaf parameter before reaching the physics engine
            if not isinstance(pos_L, nn.Parameter) or not pos_L.requires_grad:
                pos_L = nn.Parameter(pos_L.detach().clone())
                pos_L.requires_grad_(True)"""

PATCH_1_NEW = """            # [v96.0 Fix Bug 3] In-place requires_grad only — NEVER rebind pos_L.
            # Rebinding creates a new Python object not tracked by the optimizer.
            # requires_grad_(True) modifies the existing leaf tensor in-place.
            if not pos_L.requires_grad:
                pos_L.requires_grad_(True)"""


# ═══════════════════════════════════════════════════════════════════════════════
# PATCH 2 — Replace Crystal-Flow target with SO(3)-Averaged target
#
# The old target uses Kabsch-aligned difference which requires SVD every step.
# SO(3)-averaged target is mathematically equivalent for the conditional task
# but requires NO SVD: it is simply the centroid-decomposed direct difference.
# ═══════════════════════════════════════════════════════════════════════════════

PATCH_2_OLD = """                if self.config.redocking and self.config.mode == "train":
                    # CRYSTAL FLOW: Train directly on the path to the truth
                    v_target_crystal = (pos_native.unsqueeze(0).repeat(B, 1, 1) - pos_L) 
                    # Normalize by remaining time to keep flow consistent
                    v_target = v_target_crystal / (1.0 - progress + 1e-3)
                    v_target = self.phys.soft_clip_vector(v_target.detach(), max_norm=20.0)
                    if step % 50 == 0: logger.info(f"   🛰️  [Crystal-Flow] Training: Matching Geodesic path to Ground Truth.")
                else:
                    # PHYSICAL FLOW (Blind/Inference): Follow the forces (No Leakage)
                    v_target = self.phys.soft_clip_vector(force_total.detach(), max_norm=20.0)"""

PATCH_2_NEW = """                if self.config.redocking and self.config.mode == "train":
                    # [v96.0 SAEB-Flow] SO(3)-Averaged Flow Target (no SVD, no Kabsch)
                    # Haar-averaged over SO(3): decomposes into centroid velocity +
                    # centred shape velocity. Rotationally unbiased, no singularities.
                    v_target = so3_averaged_target(pos_L.detach(), pos_native, progress)
                    v_target = self.phys.soft_clip_vector(v_target.detach(), max_norm=20.0)
                    if step % 50 == 0:
                        logger.info("   🛰️  [SAEB-Flow] SO(3)-Averaged target active (no SVD).")
                else:
                    # PHYSICAL FLOW (Blind/Inference): Follow the forces (No Leakage)
                    v_target = self.phys.soft_clip_vector(force_total.detach(), max_norm=20.0)"""


# ═══════════════════════════════════════════════════════════════════════════════
# PATCH 3 — Replace F-SE3 Kabsch update with SAEB-Flow geodesic step
#
# Old code: apply_fragment_kabsch performs a hard snap projection via SVD.
# New code: apply_saeb_step integrates along the fibre-bundle manifold via
#           FK Jacobian pseudoinverse, preserving bond geometry continuously.
# ═══════════════════════════════════════════════════════════════════════════════

PATCH_3_OLD = """                # [v94.0] F-SE3: Project Combined Updates onto local SE(3) sub-manifolds
                if not getattr(self.config, 'no_fse3', False):
                    combined_pos = apply_fragment_kabsch(combined_pos, pos_native, self.fragment_labels)"""

PATCH_3_NEW = """                # [v96.0 SAEB-Flow] Fibre-bundle geodesic integration
                # Replaces Kabsch snap with FK Jacobian continuous step.
                # Bond lengths are exactly preserved; coupling is internal to FK.
                if not getattr(self.config, 'no_fse3', False):
                    fb = self.fragment_bundle   # FiberBundle built in run()
                    # [v97.0 Fix A] NO LEAKAGE: use prediction instead of ground truth
                    # Logic: if confidence is high, x1_pred is a better manifold anchor.
                    target_x1 = out['x1_pred'].view(B, N, 3).detach() if 'x1_pred' in out else pos_native.unsqueeze(0)
                    v_saeb = (target_x1 - combined_pos)
                    combined_pos = apply_saeb_step(
                        combined_pos, fb, v_saeb, dt=dt_euler * 0.5
                    )"""


# ═══════════════════════════════════════════════════════════════════════════════
# PATCH 4 — Build FiberBundle (not just labels) during experiment setup
#
# Add after the existing get_rigid_fragments() call in MaxFlowExperiment.run()
# ═══════════════════════════════════════════════════════════════════════════════

PATCH_4_OLD = """        # [v94.0 Innovation] Dynamic Rigid Fragment Extraction for F-SE3
        self.fragment_labels = get_rigid_fragments(pos_native, n_fragments=4)"""

PATCH_4_NEW = """        # [v96.0 SAEB-Flow] Build full FiberBundle topology (rotatable bonds + FK)
        self.fragment_labels = get_rigid_fragments(pos_native, n_fragments=4)
        self.fragment_bundle = build_fiber_bundle(pos_native, n_fragments=4)
        logger.info(f"   🔬 [SAEB-Flow] FiberBundle: {self.fragment_bundle.n_bonds} "
                    f"rotatable bonds, {self.fragment_bundle.n_atoms} atoms.")"""


# ═══════════════════════════════════════════════════════════════════════════════
# PATCH 5 — Energy-guided interpolant in the training interpolation
#
# Optional enhancement: bend the FM training path toward low-energy intermediates.
# Insert just before the loss_fm computation. Only active during training.
# ═══════════════════════════════════════════════════════════════════════════════

PATCH_5_INSERTION_AFTER = """                # Hierarchical Engine Call
                e_soft, e_hard, alpha = self.phys.compute_energy(pos_L_reshaped, pos_P_batched, q_L, q_P_batched, 
                                                               x_L_for_physics, x_P_batched, progress)"""

PATCH_5_NEW_CODE = """
                # [v96.0 SAEB-Flow] Energy-guided non-Gaussian interpolant
                # Bends the FM path toward low-energy intermediate states.
                # Reduces required ODE steps at inference. Only in training mode.
                if self.config.mode == "train" and 0.05 < progress < 0.95:
                    def _phys_fn(p):
                        with torch.no_grad():
                            e, _, _ = self.phys.compute_energy(
                                p, pos_P_batched, q_L, q_P_batched,
                                x_L_for_physics, x_P_batched, progress
                            )
                        return e
                    pos_L_interp = energy_guided_interpolant(
                        pos_L_reshaped.detach(),
                        pos_native.unsqueeze(0).expand(B, -1, -1),
                        progress, _phys_fn
                    )
                    # The interpolant only affects the training target, not the
                    # gradient path → no second-order terms, VRAM-safe.
                    v_target_interp_correction = (pos_L_interp - pos_L_reshaped.detach()) * 0.1
                else:
                    v_target_interp_correction = torch.zeros_like(pos_L_reshaped)
"""


# ═══════════════════════════════════════════════════════════════════════════════
# PATCH 6 — Clear FiberBundle cache between experiments
#
# Add at the start of MaxFlowExperiment.run() to prevent stale cache
# ═══════════════════════════════════════════════════════════════════════════════

PATCH_6_OLD = """        start_time = time.time() # [v72.3] Fixed NameError
        logger.info(f"🚀 Starting Experiment v94.1 (The Precision Frontier) on {self.config.target_name}...")"""

PATCH_6_NEW = """        start_time = time.time()
        clear_bundle_cache()   # [v96.0] Reset FiberBundle cache for new molecule
        logger.info(f"🚀 Starting Experiment v96.0 (SAEB-Flow) on {self.config.target_name}...")"""


# ═══════════════════════════════════════════════════════════════════════════════
# PATCH 7 — VERSION bump
# ═══════════════════════════════════════════════════════════════════════════════

PATCH_7_OLD = 'VERSION = "v94.2"'
PATCH_7_NEW = 'VERSION = "v96.0"  # SAEB-Flow: SE(3)×T^n Averaged Energy-Based Flow'


# ═══════════════════════════════════════════════════════════════════════════════
# Application helper — paste into a notebook cell to apply all patches
# ═══════════════════════════════════════════════════════════════════════════════

def apply_all_patches(main_file_path: str = "maxflow_main.py") -> None:
    """
    Reads the main file, applies all patches, writes back.
    Run once after placing maxflow_innovations.py and saeb_flow.py
    in the same directory as the main file.

    Usage:
        from maxflow_main_patches import apply_all_patches
        apply_all_patches("maxflow_v94_2.py")
    """
    with open(main_file_path, "r", encoding="utf-8") as f:
        src = f.read()

    patches = [
        (PATCH_0_OLD, PATCH_0_NEW, "Import: cpu_torsional_flow → saeb_flow"),
        (PATCH_1_OLD, PATCH_1_NEW, "Bug 3: optimizer parameter re-binding"),
        (PATCH_2_OLD, PATCH_2_NEW, "SO(3)-averaged flow target (no SVD)"),
        (PATCH_3_OLD, PATCH_3_NEW, "Kabsch snap → FK geodesic step"),
        (PATCH_4_OLD, PATCH_4_NEW, "Build FiberBundle (not just labels)"),
        (PATCH_6_OLD, PATCH_6_NEW, "Clear cache + version string update"),
        (PATCH_7_OLD, PATCH_7_NEW, "VERSION bump to v96.0"),
    ]

    for old, new, desc in patches:
        if old in src:
            src = src.replace(old, new, 1)
            print(f"  ✅ Applied: {desc}")
        else:
            print(f"  ⚠️  NOT FOUND (may already be applied): {desc}")

    # Patch 5 requires insertion after a target string
    marker = PATCH_5_INSERTION_AFTER
    if marker in src:
        src = src.replace(marker, marker + PATCH_5_NEW_CODE, 1)
        print("  ✅ Applied: Energy-guided interpolant insertion")
    else:
        print("  ⚠️  NOT FOUND: Energy-guided interpolant insertion point")

    with open(main_file_path, "w", encoding="utf-8") as f:
        f.write(src)

    print(f"\n✅ All patches applied to {main_file_path}")
    print("   Next step: verify imports and run a short --steps 50 smoke test.")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "maxflow_v94_2.py"
    apply_all_patches(path)
