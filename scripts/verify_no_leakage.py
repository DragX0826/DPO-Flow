
import os
import subprocess
import time

def verify():
    print("Verifying Data Leakage Fix...")
    
    # Create a temporary config modification to force inference mode
    test_script = "test_inference.py"
    with open(test_script, "w") as f:
        f.write("""
import torch
import logging
from lite_experiment_suite import MaxFlowExperiment, SimulationConfig

# Setup logging to capture output
logging.basicConfig(level=logging.INFO)

config = SimulationConfig(
    pdb_id="1UYD",
    target_name="1UYD_FIX_TEST",
    steps=50,
    batch_size=2,
    mode="inference",
    redocking=True # This would have leaked before
)

exp = MaxFlowExperiment(config)
exp.run()
""")

    print("   Running test inference on 1UYD...")
    try:
        # Run and capture output
        # Use python -u for unbuffered output
        import os
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        result = subprocess.run(["python", "-u", test_script], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300, env=env)
        output = result.stdout + result.stderr
        
        # Check for the magic strings
        leakage_found = "Redocking Mode: Using Ground Truth Pocket Center" in output
        fix_active1 = "[v87.1] Inference/Blind Mode: Searching for Pocket via Protein Center (No Leakage)" in output
        fix_active2 = "Follow the forces (No Leakage)" in output
        
        if leakage_found:
            print("FAILURE: Ground truth leakage detected!")
        else:
            print("SUCCESS: No ground truth leakage detected.")
            
        print("\n--- TEST LOGS ---")
        print(output.encode('ascii', errors='ignore').decode('ascii')) # Safe print
        print("-----------------\n")

        if fix_active1 and fix_active2:
            print("SUCCESS: Fix markers found in logs.")
        else:
            print("WARNING: Fix markers not found in logs.")
            if not fix_active1: print("   Missing: 'Inference/Blind Mode: Searching for Pocket via Protein Center (No Leakage)'")
            if not fix_active2: print("   Missing: 'Follow the forces (No Leakage)'")
            
        # Clean up
        if os.path.exists(test_script): os.remove(test_script)
        
    except Exception as e:
        print(f"❌ Error during verification: {e}")

if __name__ == "__main__":
    verify()
