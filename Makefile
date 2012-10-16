# Copyright (C) 2011-2012 Quantum ESPRESSO Foundation
# Copyright (C) 2010-2011 Irish Centre for High-End Computing (ICHEC)
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#
# Filippo Spiga (spiga.filippo@gmail.com)

default : prereq phigemm 

all: prereq phigemm test

prereq:
	mkdir -p ./bin ./lib ./include

phigemm:
	if test -d src ; then \
	( cd src ; if test "$(MAKE)" = "" ; then make $(MFLAGS) ; \
	else $(MAKE) $(MFLAGS) ; fi ) ; fi

test: 
	if test -d testing ; then \
	( cd testing ; if test "$(MAKE)" = "" ; then make $(MFLAGS) ; \
	else $(MAKE) $(MFLAGS) ; fi ) ; fi

clean:
	if test -d src ; then \
	( cd src ; if test "$(MAKE)" = "" ; then make $(MFLAGS) clean ; \
	else $(MAKE) $(MFLAGS) clean ; fi ) ; fi
	if test -d testing ; then \
	( cd testing ; if test "$(MAKE)" = "" ; then make $(MFLAGS) clean ; \
	else $(MAKE) $(MFLAGS) clean ; fi ) ; fi
	rm -rf ./bin ./lib

veryclean: clean
	rm make.inc
