# shiftNet
This repository contains a basic implementation of <a href="https://arxiv.org/abs/1711.08141" target="_blank"> Shift-based Resnet 20 </a> on the FPGA via High-level synthesis. 

The 'AWS' branch could run on Amazon EC2 instance. For example:
  ```
  source <path to SDSoc v2017.4>/.settings64-SDx.sh
  export SDACCEL_DIR=<path to aws-fpga>/SDAccel
  export COMMON_REPO=$SDACCEL_DIR/examples/xilinx/
  export PLATFORM=xilinx_aws-vu9p-f1_4ddr-xpr-2pr_4_0
  make TARGETS=sw_emu DEVICES=$PLATFORM all
  ```

# Quantization

The Accuracy of this network on 10000 test image from <a href="https://www.cs.toronto.edu/~kriz/cifar.html" target="_blank"> Cifar 10 </a> is 85.66% under the 8-bits weights
quantization for weights and biases. The accuracy of the original implementation
is 86.84%. 
