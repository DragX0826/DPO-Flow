import os
import shutil
import glob

def cleanup():
    base_dir = "d:/Drug"
    
    # 1. Define structure
    folders = {
        "src": ["lite_experiment_suite.py", "saeb_flow.py", "maxflow_innovations.py"],
        "scripts": ["run_benchmark_10.py", "run_ablations.py", "generate_iclr_figures.py", 
                    "create_submission.py", "verify_no_leakage.py", "benchmark_pdbbind.py",
                    "apply_fixes.py", "apply_leak_fix.py", "find_cr.py", "scan_bias.py",
                    "scientific_visualizer.py"],
        "results": ["benchmark_10_results.csv", "fig2_efficiency_pareto.pdf", 
                    "fig5_torsional_integrity.pdf", "table1_iclr_final.tex", 
                    "ablation_results.csv", "benchmark_10_results_partial.csv"],
        "archive": ["MaxFlow_v91.1_Golden_Calculus.zip", "MaxFlow_v91.2_Golden_Calculus.zip", 
                    "MaxFlow_v91.8_Golden_Calculus.zip", "MaxFlow_v94.2_Golden_Calculus.zip", 
                    "MaxFlow_v96.0_Golden_Calculus.zip", "maxflow_experiment.log"],
        "docs": ["MaxFlow_ICLR_Workshop_Draft.md", "technical_report.md", "README.md"]
    }

    # 2. Create folders
    for folder in folders:
        path = os.path.join(base_dir, folder)
        if not os.path.exists(path):
            os.makedirs(path)

    # 3. Move files
    for folder, files in folders.items():
        dest = os.path.join(base_dir, folder)
        for f in files:
            src_path = os.path.join(base_dir, f)
            if os.path.exists(src_path):
                print(f"[*] Moving {f} -> {folder}/")
                shutil.move(src_path, os.path.join(dest, f))

    # 4. Bulk cleanup patterns
    patterns = [
        "output_*.pdb", "overlay_*.pdb", "*_maxflow.pdb", "*_native.pdb",
        "view_*.pml", "ICLR_2026_Trilogy.pml", "flow_*.pdf", "fig1_*.pdf",
        "fig3_*.pdf", "fig4_*.pdf", "*.pdb", "anomaly_log.txt", "error*.log",
        "*.txt", "*.pt", "rmsd_lines.txt"
    ]
    
    for pattern in patterns:
        for f in glob.glob(os.path.join(base_dir, pattern)):
            try:
                os.remove(f)
                print(f"[-] Deleted: {os.path.basename(f)}")
            except:
                pass

    # 5. Directory cleanup
    dirs_to_remove = ["tmp_zip", "recovery", "recovery2", "upgrade", "__pycache__", "cache", "metrics"]
    for rd in dirs_to_remove:
        path = os.path.join(base_dir, rd)
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
            print(f"[-] Removed directory: {rd}")

if __name__ == "__main__":
    cleanup()
