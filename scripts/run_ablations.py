import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt
import time

def run_experiment(target, flags):
    cmd = ["python", "lite_experiment_suite.py", "--target", target, "--steps", "300", "--batch", "16", "--redocking", "--lr", "0.005", "--mode", "inference", "--b_mcmc", "64"] + flags
    print(f"[*] Running: {' '.join(cmd)}")
    my_env = os.environ.copy()
    my_env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(cmd, capture_output=True, text=True, env=my_env)
    
    # Parse RMSD from output log (logger bypasses simple subprocess stdout)
    # Searching for "MCMC Refined RMSD: X.XX A" in the log file
    rmsd = 99.9
    try:
        with open("maxflow_experiment.log", "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in reversed(lines): # Search from bottom up for efficiency
                if "MCMC Refined RMSD:" in line:
                    rmsd = float(line.split("MCMC Refined RMSD:")[1].split("A")[0].strip())
                    break
    except Exception as e:
        print(f"   [!] Log parsing failed: {e}")
    return rmsd

if __name__ == "__main__":
    target = "1UYD"
    experiments = {
        "Full (F-SE3+PI+CBSF)": [],
        "W/O Fragment-SE3": ["--no_fse3"],
        "W/O PI-Drift": ["--no_pidrift"],
        "W/O CBSF": ["--no_cbsf"],
        "Pure Cartesian (Baseline)": ["--no_fse3", "--no_pidrift", "--no_cbsf"]
    }
    
    results = {}
    for name, flags in experiments.items():
        print(f"\n[Ablation] Testing {name}...")
        rmsd = run_experiment(target, flags)
        results[name] = rmsd
        print(f"   Result: {rmsd:.2f} A")
        
    # Generate Ablation Plot
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    values = list(results.values())
    
    bars = plt.bar(names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    plt.axhline(2.0, color='red', linestyle='--', label='SOTA Barrier (2.0A)')
    plt.ylabel("RMSD (A)")
    plt.title(f"Figure 4: Component Ablation Study (Target: {target})")
    plt.xticks(rotation=15)
    plt.legend()
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}A', ha='center', va='bottom')
                
    plt.tight_layout()
    plt.savefig("fig4_ablation_results.pdf")
    print("\n[*] Ablation Study Complete. Plot saved to fig4_ablation_results.pdf")
    
    # Save results to CSV for record
    pd.DataFrame(list(results.items()), columns=["Configuration", "RMSD"]).to_csv("ablation_results.csv", index=False)
