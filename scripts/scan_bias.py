import os

def scan_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            content = f.read()
    except Exception as e:
        print(f"Failed to read {filepath}: {e}")
        return
    
    # Check for solitaire \r
    for i in range(len(content)-1):
        if content[i] == 13 and content[i+1] != 10:
            print(f"!!! Found solitaire CR at offset {i} in {filepath} !!!")
            snippet = content[max(0, i-50):min(len(content), i+150)]
            print(f"Context: {snippet}")

    # Check for suspicious math
    try:
        lines = content.decode('utf-8', errors='ignore').splitlines()
    except:
        return

    for i, line in enumerate(lines):
        cleaned = line.replace(' ', '')
        if '+1)*10' in cleaned or '+1)*10.0' in cleaned:
             print(f"!!! FOUND BIAS PATTERN !!!")
             print(f"File: {filepath}, Line {i+1}: {line.strip()}")
        
        if "'rmsd':" in line or '"rmsd":' in line:
             print(f"File: {filepath}, Line {i+1} RMSD Key: {line.strip()}")
             
        # Look for target_index + 1
        if 'idx + 1' in line or 'target_idx + 1' in line or 'index + 1' in line:
            if '* 10' in line or '*10' in line:
                print(f"File: {filepath}, Line {i+1} POSSIBLE BIAS: {line.strip()}")

if __name__ == "__main__":
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py') and file != 'scan_bias.py':
                scan_file(os.path.join(root, file))
