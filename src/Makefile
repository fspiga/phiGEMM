# Copyright (C) 2010-2011 Irish Centre for High-End Computing (ICHEC)
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#
# Filippo Spiga (filippo.spiga@ichec.ie)

include ../make.inc

PHIGEMM_MAIN_SRC = phigemm.c
#PHIGEMM_MAIN_SRC = phigemm_new.c
 
all: prereq static shared

prereq:
	mkdir -p .objs

util.o: $(PHIGEMM_TOP)/phigemm_auxiliary.c $(PHIGEMM_TOP)/phigemm_auxiliary.h $(PHIGEMM_TOP)/phigemm_common.h
	$(CC) $(CFLAGS) -c $(PHIGEMM_TOP)/phigemm_auxiliary.c $(GEMM_OPT) -o .objs/util.o $(EXT_INC) 

sgemm.o: $(PHIGEMM_TOP)/$(PHIGEMM_MAIN_SRC) $(PHIGEMM_TOP)/phigemm_auxiliary.h $(PHIGEMM_TOP)/phigemm_common.h
	$(CC) $(CFLAGS) -c $(PHIGEMM_TOP)/$(PHIGEMM_MAIN_SRC) -DCUDA_TYPE_FLOAT $(GEMM_OPT) -o .objs/sgemm.o $(EXT_INC) 

dgemm.o: $(PHIGEMM_TOP)/$(PHIGEMM_MAIN_SRC) $(PHIGEMM_TOP)/phigemm_auxiliary.h $(PHIGEMM_TOP)/phigemm_common.h
	$(CC) $(CFLAGS) -c $(PHIGEMM_TOP)/$(PHIGEMM_MAIN_SRC) -DCUDA_TYPE_DOUBLE $(GEMM_OPT) -o .objs/dgemm.o $(EXT_INC) 

cgemm.o: $(PHIGEMM_TOP)/$(PHIGEMM_MAIN_SRC) $(PHIGEMM_TOP)/phigemm_auxiliary.h $(PHIGEMM_TOP)/phigemm_common.h
	$(CC) $(CFLAGS) -c $(PHIGEMM_TOP)/$(PHIGEMM_MAIN_SRC) -DCUDA_TYPE_COMPLEX $(GEMM_OPT) -o .objs/cgemm.o $(EXT_INC)
	
zgemm.o: $(PHIGEMM_TOP)/$(PHIGEMM_MAIN_SRC) $(PHIGEMM_TOP)/phigemm_auxiliary.h $(PHIGEMM_TOP)/phigemm_common.h
	$(CC) $(CFLAGS) -c $(PHIGEMM_TOP)/$(PHIGEMM_MAIN_SRC) -DCUDA_TYPE_DOUBLE_COMPLEX $(GEMM_OPT) -o .objs/zgemm.o $(EXT_INC)

static: dgemm.o zgemm.o sgemm.o cgemm.o util.o
	mkdir -p ../bin ../lib ../include
	cp *.h ../include/.
	$(AR) $(ARFLAGS) libphigemm.a .objs/sgemm.o .objs/dgemm.o .objs/zgemm.o .objs/util.o
	mv libphigemm.a ../lib/.
	
shared: dgemm.o zgemm.o sgemm.o cgemm.o util.o
	mkdir -p ../bin ../lib ../include
	cp *.h ../include/.
	$(LD) $(LD_FLAGS) -o libphigemm.so.1.0.1 .objs/sgemm.o .objs/dgemm.o .objs/cgemm.o .objs/zgemm.o .objs/util.o $(CC_LIB)
	ln -sf libphigemm.so.1.0.1 libphigemm.so.1.0
	ln -sf libphigemm.so.1.0 libphigemm.so.1
	ln -sf libphigemm.so.1 libphigemm.so
	mv libphigemm.so* ../lib/.

clean:
	rm -rf .objs
