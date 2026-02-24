
import os

def apply_fixes(filepath):
    with open(filepath, 'r', encoding='utf-8', newline='') as f:
        content = f.read()

    # Normalize to LF for easy replacement
    c = content.replace('\r\n', '\n')

    # Fix 1: pocket_center leakage
    # Context centered around lines 932-945
    target1 = '''            use_native_center = (pos_native is not None and not self.config.blind_docking)
            
            if use_native_center and len(pos_native) > 0:
                pocket_center = pos_native.mean(0) # (3,)
                logger.info(f"   🎯 [v75.1] Redocking Mode: Using Ground Truth Pocket Center.")
            else:
                # BLIND DOCKING: Search for pocket using ESM attention interface
                logger.warning(f"   🛰️ [v75.1] Blind Docking Mode: Searching for Pocket via Protein Center...")
                pocket_center = pos_P.mean(0) 
            
            # Recenter everything on the REAL POCKET CENTER
            pos_P = pos_P - pocket_center
            pos_native = pos_native - pocket_center'''
            
    replacement1 = '''            # [v87.1] Truth-Decoupled Centering: Never use pos_native for centering during Inference
            use_native_center = (pos_native is not None and self.config.redocking and self.config.mode == "train")
            
            if use_native_center and len(pos_native) > 0:
                pocket_center = pos_native.mean(0) # (3,)
                logger.info(f"   🎯 [v87.1] Redocking Train Mode: Using Ground Truth Pocket Center.")
            else:
                # BLIND DOCKING or INFERENCE: Search for pocket or use protein center
                # [v87.1] Fixed leakage: Always use protein-based centering for inference
                logger.warning(f"   🛰️ [v87.1] Inference/Blind Mode: Searching for Pocket via Protein Center (No Leakage)...")
                pocket_center = pos_P.mean(0) 
            
            # Recenter everything on the calculated center
            pos_P = pos_P - pocket_center
            if pos_native is not None:
                pos_native = pos_native - pocket_center'''

    if target1 in c:
        c = c.replace(target1, replacement1)
        print("Applied Fix 1 (Pocket Center)")
    else:
        print("Failed to find Target 1")

    # Fix 2: v_target leakage
    # Context centered around lines 3044-3053
    target2 = '''                if self.config.redocking:
                    # CRYSTAL FLOW: Train directly on the path to the truth
                    v_target_crystal = (pos_native.unsqueeze(0).repeat(B, 1, 1) - pos_L) 
                    # Normalize by remaining time to keep flow consistent
                    v_target = v_target_crystal / (1.0 - progress + 1e-3)
                    v_target = self.phys.soft_clip_vector(v_target.detach(), max_norm=20.0)
                    if step % 50 == 0: logger.info(f"   🛰️  [Crystal-Flow] Matching Geodesic path to Ground Truth.")
                else:
                    # PHYSICAL FLOW (Blind/Inference): Follow the forces
                    v_target = self.phys.soft_clip_vector(force_total.detach(), max_norm=20.0)'''
                    
    replacement2 = '''                # [v87.1] Truth-Decoupled Dynamics: Only use Crystal Flow during Training
                if self.config.redocking and self.config.mode == "train":
                    # CRYSTAL FLOW: Train directly on the path to the truth
                    v_target_crystal = (pos_native.unsqueeze(0).repeat(B, 1, 1) - pos_L) 
                    # Normalize by remaining time to keep flow consistent
                    v_target = v_target_crystal / (1.0 - progress + 1e-3)
                    v_target = self.phys.soft_clip_vector(v_target.detach(), max_norm=20.0)
                    if step % 50 == 0: logger.info(f"   🛰️  [Crystal-Flow] Training: Matching Geodesic path to Ground Truth.")
                else:
                    # PHYSICAL FLOW (Blind/Inference): Follow the forces (No Leakage)
                    v_target = self.phys.soft_clip_vector(force_total.detach(), max_norm=20.0)'''

    if target2 in c:
        c = c.replace(target2, replacement2)
        print("Applied Fix 2 (v_target)")
    else:
        print("Failed to find Target 2")

    # Convert back to CRLF
    final_content = c.replace('\n', '\r\n')
    
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        f.write(final_content)

if __name__ == '__main__':
    apply_fixes('d:/Drug/lite_experiment_suite.py')
