# Copyright (C) 2011-2014 Quantum ESPRESSO Foundation
# Copyright (C) 2010-2011 Irish Centre for High-End Computing (ICHEC)
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#
# Filippo Spiga (filippo.spiga@quantum-espresso.org)

include ../make.inc

PHIGEMM = ..
CSCRPATH = ./serial

#OBJ = $(SRC:.f90=.o)
#EXE = $(OBJ:.o=)

# supported TEST_DATATYPE_FLAGS flags = __CUDA_TYPE_FLOAT, __CUDA_TYPE_DOUBLE, __CUDA_TYPE_COMPLEX, __CUDA_TYPE_DOUBLE_COMPLEX (one is mandatory)
TEST_DATATYPE_FLAGS = -D__CUDA_TYPE_DOUBLE

# supported TEST_OUTPUT_FLAGS flags = __CHECK_ERROR __PERFORM_PHIGEMM_INIT __PERFORM_ONLY_GPU_BIND __PERFORM_MEM_DETECT
# ** FOR DEBUGGING ONLY ** = __PHIGEMM_TESTCASE_DEBUG
# NOTE1: -D__PERFORM_PHIGEMM_INIT and -D__PHIGEMM_EXPLICIT_SPLITFACTOR are mutually exclusive
# NOTE2: if -D__PERFORM_PHIGEMM_INIT is missing, phiGEMM perform auto-init
# NOTE3: pinned memory options are -D__PHITEST_MEM_PINNED (native), -D__PHITEST_FORCE_PINNED (by API call)
TEST_FLAGS = -D__PERFORM_PHIGEMM_INIT -D__PHIGEMM_TESTCASE_DEBUG

EXTRA_TEST_FLAGS = 

TESTFLAGS = $(TEST_DATATYPE_FLAGS) $(TEST_FLAGS) $(GEMM_OPT) $(EXTRA_TEST_FLAGS)

all: prereq test

prereq:
	mkdir -p .objs

test:
	rm -rf $(PHIGEMM)/bin/*.x
	$(CC) -g $(CFLAGS) $(TESTFLAGS) -o $(PHIGEMM)/bin/single_test.x $(CSCRPATH)/single_test.c $(EXT_INC) $(LD_LIB)
	#gcc $(GEMM_OPT) -c $(CSCRPATH)/cptimer.c -o .objs/cptimer.o
	#gcc $(GEMM_OPT) -c $(CUDA_PATH)/src/fortran_thunking.c -I$(CUDA_PATH)/src $(EXT_INC) -o .objs/fortran_thunking.o
	#gcc $(GEMM_OPT) -c $(CSCRPATH)/cuda_env.c -I$(CSCRPATH) $(EXT_INC) -o .objs/cuda_env.o
	# $(FC) $(FC_PREPROC_FLAG) $(GEMM_OPT) $(TEST_FLAGS) $(EXT_INC) .objs/cuda_env.o $(CSCRPATH)/compute_split_matrix.f90 .objs/fortran_thunking.o .objs/cptimer.o $(LD_LIB) -o $(PHIGEMM)/bin/compute_split_matrix
	
run:
	./testsuite.sh

clean:
	rm -rf .objs *.o
	rm -f $(PHIGEMM)/bin/*
