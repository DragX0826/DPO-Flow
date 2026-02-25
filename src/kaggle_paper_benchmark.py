"""
SAEB-Flow FK-SMC+SOCM — Kaggle T4x2 Paper Benchmark Notebook
==============================================================
Run this notebook on Kaggle with T4x2 GPU accelerator.
It executes the 3-experiment matrix required for the 3 Claims.

Prerequisites:
  - GitHub repo attached as dataset: DragX0826/MaxFlow
  - Internet ON (for PDB download)
"""

# ── Cell 1: GPU Check ─────────────────────────────────────────────────────────
import torch
import subprocess, sys, os, time, json
import pandas as pd
import numpy as np

print("=== GPU Environment ===")
print(f"torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count:      {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

assert torch.cuda.is_available(), "ABORT: No GPU. Enable T4x2 in Kaggle settings."
assert torch.cuda.device_count() >= 1, "ABORT: Expect at least 1 GPU."

# ── Cell 2: Setup ─────────────────────────────────────────────────────────────
REPO_ROOT = "/kaggle/input/maxflow/MaxFlow"  # adjust if dataset name differs
SRC_DIR   = f"{REPO_ROOT}/src"
OUT_ROOT  = "/kaggle/working/results"
NUM_GPUS  = min(torch.cuda.device_count(), 2)

sys.path.insert(0, SRC_DIR)
os.makedirs(OUT_ROOT, exist_ok=True)

ASTEX10 = "1aq1,1b8o,1cvu,1d3p,1eve,1f0r,1fc0,1fpu,1glh,1gpk"
SEEDS   = "42,43,44"
STEPS   = 300
BATCH   = 16

print(f"SRC:     {SRC_DIR}")
print(f"NUM_GPU: {NUM_GPUS}")
print(f"Targets: {ASTEX10}")
print(f"Seeds:   {SEEDS}")

# ── Cell 3: Install deps (only once) ─────────────────────────────────────────
def shell(cmd, **kw):
    """Run shell command, print output."""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
    if r.stdout.strip(): print(r.stdout[-2000:])
    if r.returncode != 0 and r.stderr: print("STDERR:", r.stderr[-500:])
    return r.returncode

shell("pip install -q posebusters rdkit-pypi biopython esm 2>/dev/null || true")

# ── Helper ────────────────────────────────────────────────────────────────────
def run_exp(label, extra_flags, out_dir):
    """Run one benchmark config and return parsed results."""
    os.makedirs(out_dir, exist_ok=True)
    log_file = f"{out_dir}/run.log"
    t0 = time.time()
    cmd = (
        f"cd {SRC_DIR} && "
        f"python run_benchmark.py "
        f"--targets {ASTEX10} "
        f"--seeds {SEEDS} "
        f"--steps {STEPS} "
        f"--batch_size {BATCH} "
        f"--num_gpus {NUM_GPUS} "
        f"{extra_flags} "
        f"--output_dir {out_dir} "
        f"2>&1 | tee {log_file}"
    )
    print(f"\n{'='*60}")
    print(f"  [{label}] Starting experiment ...")
    print(f"{'='*60}")
    rc = shell(cmd)
    wall_time = time.time() - t0
    print(f"\n  [{label}] Done in {wall_time/60:.1f} min (exit code={rc})")

    # Parse results
    csv_path = f"{out_dir}/benchmark_aggregated.csv"
    if not os.path.exists(csv_path):
        print(f"  [WARN] No aggregated CSV for {label}!")
        return None

    df = pd.read_csv(csv_path)
    n  = len(df)
    rmsds = df["best_rmsd"].values

    # Count failures from log
    crash_count = 0
    if os.path.exists(log_file):
        with open(log_file) as f:
            crash_count = f.read().count("[FAIL]")

    # Seed variance: load per-seed CSV
    raw_csv = f"{out_dir}/benchmark_results.csv"
    seed_var = 0.0
    if os.path.exists(raw_csv):
        df_raw = pd.read_csv(raw_csv)
        if "seed" in df_raw.columns and "best_rmsd" in df_raw.columns:
            seed_sr2 = df_raw.groupby("seed").apply(
                lambda g: (g["best_rmsd"] < 2.0).mean() * 100
            )
            seed_var = seed_sr2.std()

    # ESS/resample stats (from raw CSV)
    ess_min_mean = ""
    rc_mean      = ""
    if os.path.exists(raw_csv):
        df_raw = pd.read_csv(raw_csv)
        if "ess_min" in df_raw.columns:
            ess_min_mean = f"{pd.to_numeric(df_raw['ess_min'], errors='coerce').mean():.3f}"
        if "resample_count" in df_raw.columns:
            rc_mean = f"{pd.to_numeric(df_raw['resample_count'], errors='coerce').mean():.1f}"

    return {
        "Method":        label,
        "n_targets":     n,
        "SR@2A (%)":     round((rmsds < 2.0).mean() * 100, 1),
        "SR@5A (%)":     round((rmsds < 5.0).mean() * 100, 1),
        "Median RMSD":   round(np.median(rmsds), 2),
        "Mean RMSD":     round(np.mean(rmsds), 2),
        "Crash Rate (%)": round(crash_count / (n + crash_count + 1e-8) * 100, 1),
        "Seed Var SR@2A": round(seed_var, 1),
        "ESS_min (mean)": ess_min_mean,
        "Resample/run":   rc_mean,
        "Wall Time (min)": round(wall_time / 60, 1),
        "Time/target (s)": round(wall_time / max(n, 1), 0),
    }

# ── Cell 4: Experiment Matrix ─────────────────────────────────────────────────
# Smoke test first (1 seed only, 150 steps, to confirm no crash)
print("\n" + "="*60)
print("  SMOKE TEST (1 seed, 150 steps)")
print("="*60)
shell(
    f"cd {SRC_DIR} && python run_benchmark.py "
    f"--targets 1aq1 --seed 42 --steps 150 --batch_size 8 "
    f"--num_gpus {NUM_GPUS} --fksmc --socm "
    f"--output_dir {OUT_ROOT}/smoke 2>&1 | tail -20"
)

# Main experiment matrix
EXPERIMENTS = [
    ("B_SOCM (baseline)",    "--socm",         f"{OUT_ROOT}/astex10_socm_3seed"),
    ("A_FKSMC_SOCM (ours)",  "--fksmc --socm", f"{OUT_ROOT}/astex10_fksmc_socm_3seed"),
    ("C_FKSMC_only",         "--fksmc",        f"{OUT_ROOT}/astex10_fksmc_3seed"),
]

all_results = []
for label, flags, out in EXPERIMENTS:
    res = run_exp(label, flags, out)
    if res:
        all_results.append(res)

# ── Cell 5: Summary Table ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PAPER-QUALITY SUMMARY TABLE")
print("="*70)
df_summary = pd.DataFrame(all_results)
print(df_summary.to_string(index=False))
summary_path = f"{OUT_ROOT}/paper_summary.csv"
df_summary.to_csv(summary_path, index=False)
print(f"\nSaved: {summary_path}")

# ── Cell 6: Failure Report ────────────────────────────────────────────────────
print("\n=== Failure Cases ===")
for label, flags, out in EXPERIMENTS:
    log_file = f"{out}/run.log"
    if not os.path.exists(log_file): continue
    with open(log_file) as f:
        lines = f.readlines()
    fails = [(i, l) for i, l in enumerate(lines) if "[FAIL]" in l]
    if fails:
        print(f"\n{label}:")
        for i, l in fails:
            # Print failing line + next traceback line
            print(f"  {l.strip()}")
            if i + 1 < len(lines): print(f"    -> {lines[i+1].strip()[:120]}")
    else:
        print(f"  {label}: No failures.")

# ── Cell 7: Paper Figures ─────────────────────────────────────────────────────
FIG_DIR = f"{OUT_ROOT}/figures"
os.makedirs(FIG_DIR, exist_ok=True)

try:
    sys.path.insert(0, SRC_DIR)
    from saeb.reporting.visualizer import PublicationVisualizer
    viz = PublicationVisualizer(output_dir=FIG_DIR)

    # Collect RMSD arrays for each method
    rmsd_by_method = {}
    for label, flags, out in EXPERIMENTS:
        csv_path = f"{out}/benchmark_aggregated.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            rmsd_by_method[label] = df["best_rmsd"].values

    if rmsd_by_method:
        # Fig 1: SR Curve (all methods)
        viz.plot_success_rate_curve(rmsd_by_method, filename="fig_sr_curve_comparison.pdf")
        print("Fig 1 (SR Curve) generated.")

        # Fig 2: RMSD CDF
        viz.plot_rmsd_cdf(rmsd_by_method, filename="fig_rmsd_cdf_comparison.pdf")
        print("Fig 2 (RMSD CDF) generated.")

        # Fig 4: Ablation bar chart
        ablation_data = {}
        for label, rmsds in rmsd_by_method.items():
            ablation_data[label] = {
                "sr2":         round((rmsds < 2.0).mean() * 100, 1),
                "median_rmsd": round(float(np.median(rmsds)), 2),
            }
        viz.plot_ablation(ablation_data, filename="fig_ablation_3claim.pdf")
        print("Fig 4 (Ablation) generated.")

        # Per-method benchmark summary
        for label, rmsds in rmsd_by_method.items():
            safe_label = label.replace(" ", "_").replace("/", "_")
            results_list = [{"pdb_id": f"T{i}", "best_rmsd": r} for i, r in enumerate(rmsds)]
            viz.plot_benchmark_summary(results_list, filename=f"fig_summary_{safe_label}.pdf")
        print("Fig 6 (Benchmark Summary) generated.")

    print(f"\nAll figures saved to: {FIG_DIR}")

except Exception as e:
    import traceback
    print(f"Figure generation error: {e}")
    traceback.print_exc()

# ── Cell 8: List all output files ─────────────────────────────────────────────
print("\n=== Output Files ===")
for root, dirs, files in os.walk(OUT_ROOT):
    level = root.replace(OUT_ROOT, '').count(os.sep)
    indent = '  ' * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = '  ' * (level + 1)
    for f in sorted(files):
        fsize = os.path.getsize(os.path.join(root, f)) / 1024
        print(f"{subindent}{f}  ({fsize:.0f} KB)")

# ── Cell 9: Push results to GitHub ──────────────────────────────────────────
# (Run this AFTER downloading results manually to your local machine)
# Local commands to run:
#   cp -r /kaggle/working/results/* d:/Drug/results/
#   cd d:/Drug
#   git add -A
#   git commit -m "paper-results: Astex-10 3-seed FK-SMC+SOCM benchmark v11.0"
#   git push origin main

print("""
=== DONE ===
Next steps:
1. Download /kaggle/working/results/ from Kaggle
2. Copy to local d:/Drug/results/
3. Run git commands to push to GitHub
""")
