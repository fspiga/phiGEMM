# Copyright (C) 2010-2011 Irish Centre for High-End Computing (ICHEC)
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#
# Filippo Spiga (filippo.spiga@ichec.ie)

include ../make.inc
 
all: prereq static shared fortran

prereq:
	mkdir -p .objs

util.o: $(PHIGEMM_TOP)/phigemm_auxiliary.c 
	$(NVCC) $(NVCC_FLAGS) -c $(PHIGEMM_TOP)/phigemm_auxiliary.c $(GEMM_OPT) -I../include/  -o .objs/util.o $(EXT_INC) 

sgemm.o: $(PHIGEMM_TOP)/phigemm_sgemm.c
	$(NVCC) $(NVCC_FLAGS) -c $(PHIGEMM_TOP)/phigemm_sgemm.c $(GEMM_OPT) -I../include/ -o .objs/sgemm.o $(EXT_INC) 

dgemm.o: $(PHIGEMM_TOP)/phigemm_dgemm.c
	$(NVCC) $(NVCC_FLAGS) -c $(PHIGEMM_TOP)/phigemm_dgemm.c $(GEMM_OPT) -o .objs/dgemm.o $(EXT_INC) 

cgemm.o: $(PHIGEMM_TOP)/phigemm_cgemm.c
	$(NVCC) $(NVCC_FLAGS) -c $(PHIGEMM_TOP)/phigemm_cgemm.c $(GEMM_OPT) -I../include/ -o .objs/cgemm.o $(EXT_INC)

zgemm.o: $(PHIGEMM_TOP)/phigemm_zgemm.c
	$(NVCC) $(NVCC_FLAGS) -c $(PHIGEMM_TOP)/phigemm_zgemm.c $(GEMM_OPT) -I../include/ -o .objs/zgemm.o $(EXT_INC)

dgemm_specialK.o: $(PHIGEMM_TOP)/phigemm_dgemm_specialK.c
	$(NVCC) $(NVCC_FLAGS) -c $(PHIGEMM_TOP)/phigemm_dgemm_specialK.c $(GEMM_OPT) -I../include/ -o .objs/dgemm_specialK.o $(EXT_INC)

zgemm_specialK.o: $(PHIGEMM_TOP)/phigemm_zgemm_specialK.c
	$(NVCC) $(NVCC_FLAGS) -c $(PHIGEMM_TOP)/phigemm_zgemm_specialK.c $(GEMM_OPT) -I../include/ -o .objs/zgemm_specialK.o $(EXT_INC)

static: dgemm.o zgemm.o sgemm.o cgemm.o util.o dgemm_specialK.o zgemm_specialK.o
	mkdir -p ../bin ../lib
	$(AR) $(ARFLAGS) libphigemm.a .objs/sgemm.o .objs/dgemm.o .objs/zgemm.o .objs/util.o .objs/dgemm_specialK.o .objs/zgemm_specialK.o
	mv libphigemm.a ../lib/.

shared: dgemm.o zgemm.o sgemm.o cgemm.o util.o dgemm_specialK.o zgemm_specialK.o
	mkdir -p ../bin ../lib
	$(LD) $(LD_FLAGS) $(LD_SHARED_FLAGS) -o libphigemm.so.1.0.1 .objs/sgemm.o .objs/dgemm.o .objs/cgemm.o .objs/zgemm.o .objs/util.o .objs/dgemm_specialK.o .objs/zgemm_specialK.o $(CUDA_LIB)
	ln -sf libphigemm.so.1.0.1 libphigemm.so.1.0
	ln -sf libphigemm.so.1.0 libphigemm.so.1
	ln -sf libphigemm.so.1 libphigemm.so
	mv libphigemm.so* ../lib/.

fortran:
	$(CPP) $(CPPFLAGS) $(GEMM_OPT) phigemm.f90 phigemm.F90
	$(FC) -c phigemm.F90 -o .objs/phigemm.o -module ../include/

	chmod 755 ../include/phigemm.mod

clean:
	rm -rf .objs *.F90
	rm -f ../include/*.mod
