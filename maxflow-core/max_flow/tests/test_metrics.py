from rdkit import Chem
from max_flow.utils.metrics import MultiObjectiveScorer

def test_metrics():
    print("Running Chemical Metrics Smoke Test...")
    scorer = MultiObjectiveScorer()
    
    # 1. Test with Aspirin (Known drug)
    aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    mol = Chem.MolFromSmiles(aspirin_smiles)
    
    qed = scorer.calculate_qed(mol)
    sa = scorer.calculate_sa(mol)
    reward = scorer.calculate_reward(mol)
    
    print(f"Aspirin -> QED: {qed:.4f}, SA: {sa:.4f}, Reward: {reward:.4f}")
    
    assert 0 < qed < 1, "QED should be between 0 and 1"
    assert 1 <= sa <= 10, "SA should be between 1 and 10"
    
    # 2. Test with invalid molecule
    bad_mol = None
    qed_bad = scorer.calculate_qed(bad_mol)
    print(f"Invalid Mol -> QED: {qed_bad}")
    assert qed_bad == 0.0
    
    print("✅ SUCCESS: Chemical metrics are working correctly.")

if __name__ == "__main__":
    test_metrics()
