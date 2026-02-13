import torch
from maxflow.utils.pca import compute_pca_rotation, apply_canonicalization

def test_pca_stability():
    print("Running PCA Stability Test (Sign Correction)...")
    
    # 1. Create a non-symmetric point cloud
    # We use a simple asymmetrical L-shape to ensure distinct principal axes
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.5]
    ], dtype=torch.float32)
    
    # 2. Compute canonical of original
    rot_orig, center_orig = compute_pca_rotation(pos)
    pos_can_orig = apply_canonicalization(pos, torch.zeros(5, dtype=torch.long), rot_orig, center_orig)
    
    # 3. Apply a random rotation
    # Create a random rotation matrix
    q, _ = torch.qr(torch.randn(3, 3))
    if torch.det(q) < 0:
        q[:, -1] *= -1
    
    rotated_pos = torch.matmul(pos, q.t())
    
    rot_new, center_new = compute_pca_rotation(rotated_pos)
    pos_can_new = apply_canonicalization(rotated_pos, torch.zeros(5, dtype=torch.long), rot_new, center_new)
    
    # 4. Compare
    diff = torch.abs(pos_can_orig - pos_can_new).max().item()
    print(f"Max difference after rotation: {diff:.6f}")
    
    if diff < 1e-4:
        print("✅ SUCCESS: PCA is deterministic and rotation-invariant.")
    else:
        # Debugging: show the diffs
        print("❌ FAILURE: PCA signs are still inconsistent across rotations.")
        print("Canonical Original:\n", pos_can_orig)
        print("Canonical Rotated:\n", pos_can_new)

if __name__ == "__main__":
    test_pca_stability()
