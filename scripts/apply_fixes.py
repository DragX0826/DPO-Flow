
import re

file_path = "d:/Drug/lite_experiment_suite.py"

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Clean non-breaking spaces
text = text.replace('\xa0', ' ')

# 2. Fix retain_graph=True memory leak
num_replaced = 0
text, n = re.subn(r'retain_graph\s*=\s*True', 'retain_graph=False', text)
num_replaced += n
print(f"Replaced {n} instances of retain_graph=True")

# 3. Fix pos_L.data usage 
# Regex to match pos_L.data.copy_(...) and pos_L.data += ...
def replace_copy(match):
    indent = match.group(1)
    content = match.group(2)
    return f"{indent}with torch.no_grad():\n{indent}    pos_L.copy_({content})"

def replace_add(match):
    indent = match.group(1)
    content = match.group(2)
    return f"{indent}with torch.no_grad():\n{indent}    pos_L.add_({content})"

def replace_sub(match):
    indent = match.group(1)
    content = match.group(2)
    return f"{indent}with torch.no_grad():\n{indent}    pos_L.sub_({content})"
    
def replace_basic_assign(match):
    indent = match.group(1)
    content = match.group(2)
    return f"{indent}with torch.no_grad():\n{indent}    pos_L = ({content})"

text, n1 = re.subn(r'^([ \t]*)pos_L\.data\.copy_\((.*?)\)', replace_copy, text, flags=re.MULTILINE)
text, n2 = re.subn(r'^([ \t]*)pos_L\.data\s*\+=\s*(.*)', replace_add, text, flags=re.MULTILINE)
text, n3 = re.subn(r'^([ \t]*)pos_L\.data\s*-=\s*(.*)', replace_sub, text, flags=re.MULTILINE)

# General .data assignments to pos_something.data
text, n4 = re.subn(r'(\w+)\.data\s*=\s*(.*)', r'with torch.no_grad(): \1.copy_(\2)', text)

print(f"Replaced {n1+n2+n3+n4} instances of .data assignments")

with open("d:/Drug/lite_experiment_suite_fixed.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Saved fixes to lite_experiment_suite_fixed.py")
