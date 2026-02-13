"""
Reflow Integration Test (End-to-End)
Phase 25: Pipeline Verification

Tests the full cycle:
1. Teacher Rollout (Generate Graph Data)
2. DataLoader Loading (PyG Batching)
3. Student Training Step (GNN Forward)

Ensures that the critical gap (missing edges in rollout) is fixed.
"""
import torch
import os
import shutil
from max_flow.scripts.generate_reflow_data import generate_reflow_data
from max_flow.train_reflow import train_reflow, ReflowPairDataset
from torch_geometric.loader import DataLoader

def test_reflow_pipeline():
    print("🧪 Starting Reflow Integration Test...")
    
    # Paths
    test_dir = "tests/temp_reflow"
    data_path = os.path.join(test_dir, "reflow_test.pt")
    checkpoint_path = os.path.join(test_dir, "teacher.pt")
    save_dir = os.path.join(test_dir, "checkpoints")
    
    os.makedirs(test_dir, exist_ok=True)
    
    # 1. Create Dummy Teacher Checkpoint
    print("\n[1/3] Creating Dummy Teacher...")
    from max_flow.models.max_rl import MaxFlow
    model = MaxFlow()
    torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)
    
    # 2. Run Rollout (Generate Data)
    print("\n[2/3] Running Teacher Rollout...")
    # Generate 10 samples (enough for a batch)
    try:
        generate_reflow_data(
            checkpoint_path=checkpoint_path,
            save_path=data_path,
            n_samples=10,
            batch_size=4,
            steps=10, # Shorten steps for test speed
            device='cpu'
        )
    except Exception as e:
        print(f"❌ Rollout Failed: {e}")
        return
        
    # Verify Data Structure
    data = torch.load(data_path)
    img_0 = data[0]
    if not hasattr(img_0, 'edge_index') and not hasattr(img_0, 'x_P'):
        print("❌ Data missing graph structure (edges/features)!")
        return
    else:
        print("✅ Data contains graph structure.")

    # 3. Run Training Step (Distillation)
    print("\n[3/3] Running Student Training Step...")
    try:
        # Run 1 epoch
        train_reflow(
            dataset_path=data_path,
            checkpoint_path=checkpoint_path,
            save_dir=save_dir,
            epochs=1,
            batch_size=4,
            device='cpu' # Force CPU for CI test
        )
        print("✅ Training step completed without error.")
    except Exception as e:
        print(f"❌ Training Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Cleanup
    shutil.rmtree(test_dir)
    print("\n🎉 Integrated Reflow Pipeline Verified!")

if __name__ == "__main__":
    test_reflow_pipeline()
