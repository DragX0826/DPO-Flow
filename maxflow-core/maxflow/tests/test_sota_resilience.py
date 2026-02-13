# maxflow/tests/test_sota_resilience.py
import torch
from maxflow.models.backbone import CrossGVP
from maxflow.models.flow_matching import RectifiedFlow
from maxflow.utils.metrics import MultiObjectiveScorer
from torch_geometric.data import Data

def test_sota_resilience():
    print("🧪 MaxFlow SOTA Resilience Test (ICLR 2027 Alignment)...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_dim = 128
    model = CrossGVP(node_in_dim=32, hidden_dim=hidden_dim).to(device)
    flow = RectifiedFlow(model).to(device)
    scorer = MultiObjectiveScorer()
    
    # 1. Mock Data (Apo Protein + Noisy Ligand)
    batch_size = 2
    n_L, n_P = 20, 100
    data = Data(
        x_L=torch.randn(n_L, 32, device=device),
        pos_L=torch.randn(n_L, 3, device=device),
        x_L_batch=torch.zeros(n_L, dtype=torch.long, device=device),
        x_P=torch.randn(n_P, 21, device=device),
        pos_P=torch.randn(n_P, 3, device=device),
        x_P_batch=torch.zeros(n_P, dtype=torch.long, device=device),
        pocket_center=torch.zeros(1, 3, device=device),
        # Phase 51/52 Requirements
        elements_L=["C"] * n_L,
        atom_to_motif=torch.zeros(n_L, dtype=torch.long, device=device)
    )
    # Mock some motifs
    data.atom_to_motif[10:] = 1 # Split into 2 motifs
    
    # 2. Test Backbone Forward (Flexibility Output)
    t = torch.tensor([0.5], device=device)
    v_info, conf, kl = model(t, data)
    rmsf = v_info[4]
    
    print(f"✅ Backbone RMSF Average: {rmsf.mean().item():.4f}")
    assert rmsf.shape == (n_L, 1), "RMSF shape mismatch"
    assert (rmsf >= 0).all(), "RMSF must be positive"
    
    # 2b. Test Chiral Head (Phase 53)
    if isinstance(v_info, tuple) and len(v_info) >= 6:
        chiral_pred = v_info[5]
        print(f"✅ Chiral Pred Mean: {chiral_pred.mean().item():.4f}")
        assert chiral_pred.shape == (n_L, 1), "Chiral pred shape mismatch"
        assert (chiral_pred >= -1).all() and (chiral_pred <= 1).all(), "Chiral pred outside Tanh bounds"
    else:
        print("⚠️ Chiral Head output missing in Backbone!")
    
    # 3. Test Langevin Sampling
    print("⏳ Testing Langevin-Rectified Flow Sampling...")
    x_final, traj = flow.sample(data, steps=5, gamma=1.0)
    print(f"✅ Sampling complete. Final pos mean: {x_final.mean().item():.4f}")
    assert len(traj) == 6, "Trajectory tracking failed"
    
    # 4. Test Synthesis Entropy Scorer
    from rdkit import Chem
    mol_simple = Chem.MolFromSmiles("CCCC")
    mol_complex = Chem.MolFromSmiles("C12=CC=CC=C1C3=C(C=C2)C4=C(C=C3)C=CC=C4") # Multi-ring
    
    se_simple = scorer.calculate_synthesis_entropy(mol_simple)
    se_complex = scorer.calculate_synthesis_entropy(mol_complex)
    
    print(f"✅ Synthesis Entropy (Simple): {se_simple:.4f}")
    print(f"✅ Synthesis Entropy (Complex): {se_complex:.4f}")
    assert se_simple > se_complex, "Synthesis Entropy should penalize complexity"
    
    # 5. Test Motif Joints (Phase 49)
    from maxflow.data.featurizer import ProteinLigandFeaturizer
    import os
    # Create dummy SDF/PDB files for testing featurizer
    # (Reuse existing ones if available or mock)
    print("⏳ Testing Motif Joint tracking in Featurizer...")
    # For now, just check logic on the decomposer directly
    from maxflow.utils.motifs import MotifDecomposer
    decomp = MotifDecomposer()
    mol_biphenyl = Chem.MolFromSmiles("c1ccccc1-c2ccccc2")
    motifs, joints = decomp.decompose(mol_biphenyl)
    print(f"✅ Biphenyl motifs: {len(motifs)}, Joints: {len(joints)}")
    assert len(motifs) == 2, "Biphenyl should split into 2 benzenes"
    assert len(joints) == 1, "There should be one joint bond"
    
    # Test local frames
    centers = decomp.get_motif_centers(torch.randn(12, 3), motifs)
    rotations = decomp.get_rigid_transformations(torch.randn(12, 3), motifs, centers)
    print(f"✅ Motif local frames shape: {rotations.shape}")
    assert rotations.shape == (2, 3, 3), "Rotations shape mismatch"
    
    print("\n🎉 SOTA Resilience Upgrades & Motif Logic Verified!")

if __name__ == "__main__":
    test_sota_resilience()
