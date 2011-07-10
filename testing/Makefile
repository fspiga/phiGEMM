# Copyright (C) 2010-2011 Irish Centre for High-End Computing (ICHEC)
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#
# Filippo Spiga (filippo.spiga@ichec.ie)

include ../make.inc

PHIGEMM = ..
CSCRPATH = ./serial

OBJ = $(SRC:.f90=.o)
EXE = $(OBJ:.o=)

# supported TEST_DATATYPE_FLAGS flags = __CUDA_TYPE_FLOAT, __CUDA_TYPE_DOUBLE, __CUDA_TYPE_COMPLEX, __CUDA_TYPE_DOUBLE_COMPLEX (one is mandatory)
TEST_DATATYPE_FLAGS = -D__CUDA_TYPE_DOUBLE

# supported TEST_OUTPUT_FLAGS flags = __LONG_OUTPUT, __CHECK_ERROR __PERFORM_PHIGEMM_INIT
TEST_FLAGS = -D__PERFORM_PHIGEMM_INIT

TESTFLAGS = $(TEST_DATATYPE_FLAGS) $(TEST_FLAGS) $(GEMM_OPT)

all: prereq test

prereq:
	mkdir -p .objs

test: test_c 

run: run_c

test_c:
	rm -f $(PHIGEMM)/bin/*.x
	$(CC) $(CFLAGS) $(TESTFLAGS) -o $(PHIGEMM)/bin/single_test.x $(CSCRPATH)/single_test.c $(EXT_INC) -I$(PHIGEMM)/include -L$(PHIGEMM)/lib -lphigemm $(LD_LIB)

run_c:
	env LD_LIBRARY_PATH=../lib:${LD_LIBRARY_PATH} numactl -m 1 -c 1 ../bin/comparative_test.x

clean:
	rm -rf .objs
	rm -f $(PHIGEMM)/bin/*
