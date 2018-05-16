COMMON_DIR:=params/

#Common Includes
include $(COMMON_REPO)/utility/boards.mk
include $(COMMON_REPO)/libs/xcl2/xcl2.mk
include $(COMMON_REPO)/libs/opencl/opencl.mk

# Host Application
run_bs_SRCS=main_cl.cpp $(oclHelper_SRCS) $(xcl2_SRCS)
run_bs_HDRS=$(xcl2_HDRS)
run_bs_CXXFLAGS=-I$(COMMON_DIR) $(opencl_CXXFLAGS) $(xcl2_CXXFLAGS) $(oclHelper_CXXFLAGS) -std=c++0x
run_bs_LDFLAGS=$(opencl_LDFLAGS)

EXES=run_bs

# Kernel
shift_SRCS=shift.cpp 

KERNEL_NAME:=shift
XOS=$(KERNEL_NAME)

shift_XOS=$(KERNEL_NAME)

XCLBINS=$(KERNEL_NAME)
CLFLAGS=-I$(COMMON_DIR) -I. --kernel $(KERNEL_NAME) --xp "param:compiler.preserveHlsOutput=1" --xp "param:compiler.generateExtraRunData=true" -s 

# check
check_EXE=run_bs
check_XCLBINS=$(KERNEL_NAME) 

CHECKS=check

include $(COMMON_REPO)/utility/rules.mk

