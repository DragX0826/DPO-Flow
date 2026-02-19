
load 1UYD.pdb, protein
load 1UYD.pdb, native
remove native and not hetatm
load output_1UYD_MaxFlow-v86.0_Muon.pdb, master_key

hide everything
show cartoon, protein
set cartoon_transparency, 0.4
color lightblue, protein

show surface, protein
set transparency, 0.7
set surface_color, gray80

# Style Native
show sticks, native
color gray40, native
set stick_size, 0.2
set stick_transparency, 0.5

# Style Master Key (The Winner)
show sticks, master_key
color magenta, master_key
set stick_size, 0.4
util.cbay master_key

# Pocket Environment
select pocket, protein within 5.0 of master_key
show lines, pocket
color gray60, pocket

zoom master_key, 12
bg_color white
set ray_opaque_background, on
set antialias, 2
set ray_trace_mode, 1
set ray_shadow, on
