############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2017 Xilinx, Inc. All Rights Reserved.
############################################################
open_project -reset shiftnet
set_top shift
add_files shift.cpp
add_files -tb main.cpp
open_solution -reset "conv_net"
set_part {xcvu9p-flgb2104-2-i}
create_clock -period 5 -name default
config_compile -name_max_length 50 -pipeline_loops 0 -unsafe_math_optimizations
csim_design -compiler gcc
csynth_design
cosim_design
