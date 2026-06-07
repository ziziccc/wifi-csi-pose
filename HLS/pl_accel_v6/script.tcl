open_project -reset pl_accel_v6
set_top full_pose_accel

add_files full_pose.cpp
add_files full_pose.h
add_files -tb testbench.cpp
add_files -tb test_vectors.h

open_solution -reset "sol1"
set_part {xc7z020clg400-1}
create_clock -period 7.5 -name default

csim_design
csynth_design
# cosim_design
# export_design -format ip_catalog

exit
