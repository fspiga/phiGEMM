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
	$(PHIGEMM_NVCC) $(PHIGEMM_NVCC_FLAGS) $(PHIGEMM_GEMM_OPT) $(PHIGEMM_EXT_INC) -I../include/ -c $< -o $*.o 

include ../make.inc
 
default: all
 
all: static shared fortran

PHIGEMM_OBJS= \
phigemm_auxiliary.o \
phigemm_env.o \
phigemm_dgemm.o \
phigemm_dgemm_specialK.o \
phigemm_zgemm.o \
phigemm_zgemm_specialK.o

static: $(PHIGEMM_OBJS)
	mkdir -p ../bin ../lib
	$(PHIGEMM_AR) $(PHIGEMM_ARFLAGS) libphigemm.a ${PHIGEMM_OBJS}
	mv libphigemm.a ../lib/.

shared: $(PHIGEMM_OBJS)
	mkdir -p ../bin ../lib
	$(PHIGEMM_LD) $(PHIGEMM_LD_FLAGS) $(PHIGEMM_LD_SHARED_FLAGS) -o libphigemm.so.2.0 $(PHIGEMM_OBJS) $(PHIGEMM_CUDA_LIB) $(PHIGEMM_LD_LIB)
	ln -sf libphigemm.so.2.0 libphigemm.so.2
	ln -sf libphigemm.so.2 libphigemm.so
	mv libphigemm.so* ../lib/.

fortran:
	$(PHIGEMM_CPP) $(PHIGEMM_CPPFLAGS) $(PHIGEMM_GEMM_OPT) phigemm.f90 phigemm.F90
	$(PHIGEMM_FC) $(PHIGEMM_FFLAGS) -c phigemm.F90 -o phigemm.o
	chmod 755 ../include/*.mod

clean:
	rm -rf *.o *.F90
	rm -f ../include/*.mod

include make.depend
	
