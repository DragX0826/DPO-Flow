import sys

with open('lite_experiment_suite.py', 'rb') as f:
    data = f.read()

for i in range(len(data)-1):
    if data[i] == 13 and data[i+1] != 10:
        print(f"Solitary CR at offset {i}")
        snippet = data[max(0, i-40):min(len(data), i+100)].decode('utf-8', errors='ignore')
        print(f"Context: {repr(snippet)}")
