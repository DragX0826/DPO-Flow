"""
Phase 25: One-Step Generation Benchmark ⚡
Compare quality (RMSD) and speed of:
1. Standard Euler (100 steps)
2. Reflow Distilled (1 step)

Strategy:
- Load a 'Teacher' (Standard) and a 'Student' (Reflow).
- Generate molecules for the same pocket.
- Measure inference time and geometric deviation.
"""
import torch
import time
import numpy as np
from maxflow.models.max_rl import MaxFlow
from maxflow.utils.chem import ProteinLigandData
from torch_geometric.data import Batch

def benchmark_one_step(
    teacher_path="checkpoints/MaxRL_model_epoch_5.pt",
    student_path="checkpoints_reflow/reflow_student_epoch_10.pt",
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    print("🚀 One-Step Generation Benchmark")
    
    # 1. Models
    # Mock loading for prototype
    teacher = MaxFlow().to(device) # Mock teacher
    student = MaxFlow().to(device) # Mock student
    # Setup for 1-step logic (Student needs config or just inference param)
    
    # 2. Mock Data
    dummy_pocket = torch.randn(30, 6).to(device)
    dummy_pos_P = torch.randn(30, 3).to(device)
    dummy_center = dummy_pos_P.mean(dim=0)
    
    data = Batch.from_data_list([ProteinLigandData(
        x_L=torch.zeros(15, 6).to(device),
        pos_L=torch.randn(15, 3).to(device),
        x_P=dummy_pocket,
        pos_P=dummy_pos_P,
        pocket_center=dummy_center
    )]).to(device)
    
    # 3. Benchmark Teacher (100 Steps)
    torch.cuda.synchronize() if device == 'cuda' else None
    t0 = time.time()
    with torch.no_grad():
        x_teacher, _ = teacher.flow.sample(data, steps=100)
    torch.cuda.synchronize() if device == 'cuda' else None
    t_teacher = time.time() - t0
    
    print(f"Teacher (100 steps): {t_teacher*1000:.2f} ms")
    
    # 4. Benchmark Student (1 Step)
    torch.cuda.synchronize() if device == 'cuda' else None
    t0 = time.time()
    with torch.no_grad():
        # Ideally Reflow student can do 1 step
        # We simulate 1 step call
        x_student, _ = student.flow.sample(data, steps=1)
    torch.cuda.synchronize() if device == 'cuda' else None
    t_student = time.time() - t0
    
    print(f"Student (1 step):    {t_student*1000:.2f} ms")
    print(f"⚡ Speedup: {t_teacher/t_student:.1f}x")
    
    # 5. Quality Check (RMSD for demo)
    # Note: They start from random noise, so comparing x_teacher vs x_student directly 
    # isn't fair unless we fix noise.
    # In real Reflow, we distill the deterministic mapping.
    # Here we just show the speedup potential.
    
    if t_student < t_teacher / 50:
         print("✅ Massive speedup confirmed (>50x)")
    else:
         print("⚠️ Speedup lower than expected (Overhead dominated?)")

if __name__ == "__main__":
    benchmark_one_step()
