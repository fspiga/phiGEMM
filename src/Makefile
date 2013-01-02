# Copyright (C) 2011-2012 Quantum ESPRESSO Foundation
# Copyright (C) 2010-2011 Irish Centre for High-End Computing (ICHEC)
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#
# Filippo Spiga (spiga.filippo@gmail.com)

include ../make.inc
 
default: all
 
all: static shared fortran

PHIGEMM_OBJS= \
phigemm_auxiliary.o \
phigemm_env.o \
phigemm_dgemm.o \
phigemm_zgemm.o \
phigemm_dgemm_specialK.o \
phigemm_zgemm_specialK.o \
phigemm_cgemm.o 
#\
#phigemm_sgemm.o

static: $(PHIGEMM_OBJS)
	mkdir -p ../bin ../lib
	$(AR) $(ARFLAGS) libphigemm.a ${PHIGEMM_OBJS}
	mv libphigemm.a ../lib/.

shared: $(PHIGEMM_OBJS)
	mkdir -p ../bin ../lib
	$(LD) $(LD_FLAGS) $(LD_SHARED_FLAGS) -o libphigemm.so.2.0 $(PHIGEMM_OBJS) $(CUDA_LIB) $(LD_LIB)
	ln -sf libphigemm.so.2.0 libphigemm.so.2
	ln -sf libphigemm.so.2 libphigemm.so
	mv libphigemm.so* ../lib/.

fortran:
	$(CPP) $(CPPFLAGS) $(GEMM_OPT) phigemm.f90 phigemm.F90
	$(FC) $(FFLAGS) -c phigemm.F90 -o phigemm.o

	chmod 755 ../include/phigemm.mod

clean:
	rm -rf *.o *.F90
	rm -f ../include/*.mod
