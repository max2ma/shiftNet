# shiftNet
This repository contains a basic implementation of  <a href="https://arxiv.org/abs/1711.08141" target="_blank"> Shift-based Resnet </a> on the FPGA via High-level synthesis. 

The 'AWS' branch could run on Amazon EC2 instance. For example:
  ```
  source <path to SDSoc v2017.1>/.settings64-SDx.sh
  export SDACCEL_DIR=<path to aws-fpga>/SDAccel
  export COMMON_REPO=$SDACCEL_DIR/examples/xilinx/
  export PLATFORM=xilinx_aws-vu9p-f1_4ddr-xpr-2pr_4_0
  make TARGETS=sw_emu DEVICES=$PLATFORM all
  ```
