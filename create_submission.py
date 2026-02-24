import os
import zipfile
import shutil
from datetime import datetime

def create_submission():
    """
    Packages all ICLR Workshop submission artifacts into a single ZIP file.
    """
    submission_name = f"maxflow_iclr_submission_v97.6"
    zip_filename = f"{submission_name}.zip"
    
    # Files to include
    files_to_include = [
        "d:/Drug/docs/MaxFlow_ICLR_Workshop_Draft.md",
        "d:/Drug/README.md",
        "d:/Drug/src/lite_experiment_suite.py",
        "d:/Drug/src/saeb_flow.py",
        "d:/Drug/src/saebflow_innovations.py",
        "d:/Drug/results/fig2_efficiency_pareto.pdf",
        "d:/Drug/results/fig5_torsional_integrity.pdf",
        "d:/Drug/results/benchmark_10_results.csv",
        "C:/Users/user/.gemini/antigravity/brain/b923002c-cbaf-47d8-8167-f8ae5b7fa0f0/walkthrough_v97.6.md",
        "C:/Users/user/.gemini/antigravity/brain/b923002c-cbaf-47d8-8167-f8ae5b7fa0f0/task.md"
    ]
    
    # We are on Windows, ensure paths are correct
    files_to_include = [f.replace("/", "\\") for f in files_to_include]
    
    print(f"[*] Creating submission package: {zip_filename}")
    
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for file_path in files_to_include:
            if os.path.exists(file_path):
                # Use basename for the file in the zip
                arcname = os.path.basename(file_path)
                zipf.write(file_path, arcname)
                print(f"  [+] Added: {arcname}")
            else:
                print(f"  [!] Warning: {file_path} not found. Skipping.")
                
    print(f"\n[SUCCESS] Submission package created at: {os.path.abspath(zip_filename)}")

if __name__ == "__main__":
    create_submission()
