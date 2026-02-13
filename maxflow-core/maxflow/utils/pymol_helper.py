# maxflow/utils/pymol_helper.py

import os

def generate_pml_script(output_dir, protein_name="protein.pdb", ligand_name="ligand.sdf", script_name="show_pose.pml"):
    """
    Creates a PyMOL script (.pml) for high-quality figure rendering.
    """
    pml_content = f"""
# Load structures
load {protein_name}, protein
load {ligand_name}, ligand

# Style Protein
hide everything, protein
show cartoon, protein
color grey80, protein
set cartoon_transparency, 0.5

# Style Ligand
show sticks, ligand
color green, ligand
set stick_radius, 0.2

# Pocket Surface
select pocket, (protein within 8.0 of ligand)
show surface, pocket
set transparency, 0.7
color lightblue, pocket

# View setup
center ligand
zoom ligand, 10
set bg_rgb, [1, 1, 1]
set ray_opaque_background, on

# Action
orient ligand
"""
    script_path = os.path.join(output_dir, script_name)
    with open(script_path, "w") as f:
        f.write(pml_content)
    
    return script_path

def create_multi_pose_pml(output_dir, protein_file, ligand_files, script_name="gallery.pml"):
    """
    Creates a PyMOL gallery script to compare several generated poses.
    """
    lines = [f"load {protein_file}, protein", "hide everything, protein", "show cartoon, protein", "color gray90, protein"]
    
    colors = ["cyan", "magenta", "orange", "yellow", "marine"]
    for i, l_file in enumerate(ligand_files):
        name = f"pose_{i}"
        color = colors[i % len(colors)]
        lines.append(f"load {l_file}, {name}")
        lines.append(f"show sticks, {name}")
        lines.append(f"color {color}, {name}")
    
    lines.append("center protein")
    lines.append("zoom pose_0, 10")
    
    script_path = os.path.join(output_dir, script_name)
    with open(script_path, "w") as f:
        f.write("\n".join(lines))
    
    return script_path
