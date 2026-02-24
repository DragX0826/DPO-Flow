
load 1UYD.pdb, protein
load 1UYD.pdb, native
remove native and not hetatm

# Load Trajectory Snapshots
load output_step0.pdb, step0
load output_step200.pdb, step200
load output_1UYD_SAEB-Flow_Muon.pdb, final_refined

hide everything
show cartoon, protein
set cartoon_transparency, 0.4
color lightblue, protein

show surface, protein
set transparency, 0.8
set surface_color, gray90

# Style Native (Reference)
show sticks, native
color gray30, native
set stick_size, 0.2
set stick_transparency, 0.6
show spheres, native
set sphere_scale, 0.2, native

# Style Trajectory Evolution
# Step 0: Pale and thin (The 'Cloud' state)
show sticks, step0
color yellow, step0
set stick_size, 0.15
set stick_transparency, 0.5

# Step 200: Intermediate transition
show sticks, step200
color orange, step200
set stick_size, 0.25
set stick_transparency, 0.3
show spheres, step200
set sphere_scale, 0.2, step200

# Final: Magenta and thick (The 'Lock-in' state)
show sticks, final_refined
color magenta, final_refined
set stick_size, 0.45
# Professional styling for Final Pose
show spheres, final_refined
set sphere_scale, 0.3
# Ensure we see SOMETHING even if bonds fail
show nb_spheres, final_refined
# Professional styling for Final Pose
show spheres, final_refined
set sphere_scale, 0.3
# Ensure we see SOMETHING even if bonds fail
show nb_spheres, final_refined

# Pocket Environment (5.0A around Final Pose)
select pocket, protein within 5.0 of final_refined
show lines, pocket
color gray50, pocket

zoom final_refined, 15
bg_color white
set ray_opaque_background, on
set antialias, 2
set ray_trace_mode, 1
set ray_shadow, on
