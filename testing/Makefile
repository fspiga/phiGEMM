# Copyright (C) 2011-2014 Quantum ESPRESSO Foundation
# Copyright (C) 2010-2011 Irish Centre for High-End Computing (ICHEC)
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#
# Filippo Spiga (filippo.spiga@quantum-espresso.org)

.SUFFIXES :
.SUFFIXES : .o .c
.c.o:
	$(NVCC) $(NVCC_FLAGS) $(GEMM_OPT) $(EXT_INC) -I../include/ -c $< -o $*.o 

include ../make.inc


# supported TEST_DATATYPE_FLAGS flags = __CUDA_TYPE_DOUBLE, __CUDA_TYPE_DOUBLE_COMPLEX (one is mandatory)
TEST_DATATYPE_FLAGS = -D__CUDA_TYPE_DOUBLE

# supported TEST_OUTPUT_FLAGS flags = __CHECK_ERROR __PERFORM_PHIGEMM_INIT __PERFORM_ONLY_GPU_BIND __PERFORM_MEM_DETECT
# ** FOR DEBUGGING ONLY ** = __PHIGEMM_TESTCASE_DEBUG
# NOTE1: -D__PERFORM_PHIGEMM_INIT and -D__PHIGEMM_EXPLICIT_SPLITFACTOR are mutually exclusive
# NOTE2: if -D__PERFORM_PHIGEMM_INIT is missing, phiGEMM perform auto-init
# NOTE3: pinned memory options are -D__PHITEST_MEM_PINNED (native), -D__PHITEST_FORCE_PINNED (by API call)
TEST_FLAGS = -D__PERFORM_PHIGEMM_INIT -D__PHIGEMM_TESTCASE_DEBUG -D__CHECK_ERROR

TESTFLAGS = $(TEST_DATATYPE_FLAGS) $(TEST_FLAGS) $(GEMM_OPT) $(EXTRA_TEST_FLAGS)

all: prereq test

prereq:
	mkdir -p .objs

test:
	rm -rf $(TOPDIR)/bin/*.x
	$(CC) -g $(CFLAGS) $(TESTFLAGS) -o $(TOPDIR)/bin/single_test.x single_test.c $(EXT_INC) $(LD_LIB)
	
run:
	./testsuite.sh

clean:
	rm -rf .objs *.o
	rm -f $(TOPDIR)/bin/*
