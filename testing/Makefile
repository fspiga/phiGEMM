# Copyright (C) 2011-2015 Quantum ESPRESSO Foundation
# Copyright (C) 2010-2011 Irish Centre for High-End Computing (ICHEC)
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#
# Filippo Spiga (filippo.spiga@quantum-espresso.org)

include ../make.inc

PHIGEMM_TEST_FLAGS = $(PHIGEMM_GEMM_OPT)
PHIGEMM_TEST_FLAGS += -D__PHITEST_PHIGEMM_INIT 
#PHIGEMM_TEST_FLAGS += -D__PHITEST_MEM_PINNED
#PHIGEMM_TEST_FLAGS += -D__PHITEST_FORCE_PINNED
PHIGEMM_TEST_FLAGS += -D__PHITEST_TESTCASE_DEBUG
PHIGEMM_TEST_FLAGS += -D__PHITEST_CHECK_ERROR
#PHIGEMM_TEST_FLAGS += -D__PHITEST_PERFORM_ONLY_GPU_BIND
#PHIGEMM_TEST_FLAGS += -D__PHITEST_PERFORM_MEM_DETECT

all: prereq test

prereq:
	mkdir -p .objs

test:
	$(PHIGEMM_CC) -g $(PHIGEMM_CFLAGS) -D__CUDA_TYPE_DOUBLE $(PHIGEMM_TEST_FLAGS) -I../include $(PHIGEMM_EXT_INC) -o ../bin/single_test-dgemm.x single_test.c ../lib/libphigemm.a $(PHIGEMM_LD_LIB)
	$(PHIGEMM_CC) -g $(PHIGEMM_CFLAGS) -D__CUDA_TYPE_DOUBLE_COMPLEX $(PHIGEMM_TEST_FLAGS) -I../include $(PHIGEMM_EXT_INC) -o ../bin/single_test-zgemm.x single_test.c $(PHIGEMM_LD_LIB)
	
run:
	./testsuite.sh

clean:
	rm -rf .objs *.o
	rm -f ../bin/*
