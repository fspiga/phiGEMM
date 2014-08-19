/*
 * Copyright (C) 2011-2014 Quantum ESPRESSO Foundation
 * Copyright (C) 2010-2011 Irish Centre for High-End Computing (ICHEC)
 *
 * This file is distributed under the terms of the
 * GNU General Public License. See the file `License'
 * in the root directory of the present distribution,
 * or http://www.gnu.org/copyleft/gpl.txt .
 *
 * Filippo Spiga (filippo.spiga@quantum-espresso.org)
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>

#include <time.h>
#include <sys/types.h>
#include <sys/time.h>

#include "phigemm.h"
#include "phigemm_auxiliary.h"

#if defined(__PHIGEMM_PROFILE)
const char base[] = "phigemm.profile";
#endif

void readEnv(int tag)
{
	/*
	 * <phiGEMM data structure>.<field>       --> env variable
	 *
	 * myPhiGemmHdl.split  					  --> PHI_GEMM_SPLIT
	 * myPhiGemmHdl.SPLITK_FACTOR             --> PHI_SPLITK_FACTOR
	 * myPhiGemmHdl.SPLITK                    --> PHI_SPLITK_GEMM
	 * myPhiGemmHdl.LOWER_LIMIT               --> PHI_LOWER_LIMIT
	 * myPhiGemmHdl.UPPER_LIMIT_NM            --> PHI_UPPER_LIMIT_NM
	 * myPhiGemmHdl.UPPER_LIMIT_K             --> PHI_UPPER_LIMIT_K
	 *
	 * myPhiGemmEnv.filename                  --> base + PHI_PROFILE_PREFIX
	 *
	 */

	float envar;
	char *value = NULL;

	/* PHI_GEMM_SPLIT */
	value = getenv("PHI_GEMM_SPLIT");
	if (value != NULL)
	{
		envar = atof(value);
	} else {
		/* Default if no env variable is specified */
		envar = 0.95;
	}
	myPhiGemmHdl.SPLIT = envar;


	/* SPLITK_FACTOR */
	value = getenv("PHI_SPLITK_FACTOR");
	if (value != NULL)
	{
		envar = atof(value);

	} else {
		/* Default if no env variable is specified */
		envar = __SPLITK_FACTOR;
	}
	myPhiGemmHdl.SPLITK_FACTOR = envar;

	/* SPLITK_GEMM */
	value = getenv("PHI_SPLITK_GEMM");
	if (value != NULL)
	{
		envar = atoi(value);

	} else {
		/* Default if no env variable is specified */
		envar = __SPLITK_GEMM;
	}
	myPhiGemmHdl.SPLITK_GEMM = envar;

	/* LOWER_LIMIT */
	value = getenv("PHI_LOWER_LIMIT");
	if (value != NULL)
	{
		envar = atoi(value);
	} else {
		/* Default if no env variable is specified */
		envar = __LOWER_LIMIT;
	}
	myPhiGemmHdl.LOWER_LIMIT = envar;


	/* UPPER_LIMIT_NM */
	value = getenv("PHI_UPPER_LIMIT_NM");
	if (value != NULL)
	{
		envar = atoi(value);
	} else {
		/* Default if no env variable is specified */
		envar = __UPPER_LIMIT_NM;
	}
	myPhiGemmHdl.UPPER_LIMIT_NM = envar;


	/* UPPER_LIMIT_K */
	value = getenv("PHI_UPPER_LIMIT_K");
	if (value != NULL)
	{
		envar = atoi(value);
	} else {
		/* Default if no env variable is specified */
		envar = __UPPER_LIMIT_K;
	}
	myPhiGemmHdl.UPPER_LIMIT_K = envar;

#if defined(__PHIGEMM_PROFILE)
	/* Create file descriptor where store the profiling information */

	value = getenv("PHI_PROFILE_PREFIX");

	if (tag < 0) {
		if (value != NULL)
			sprintf(myPhiGemmHdl.filename, "%s.%s.csv", base, value);
		else
			sprintf(myPhiGemmHdl.filename, "%s.csv", base);
	} else {
		if (value != NULL)
			sprintf(myPhiGemmHdl.filename, "%s.%d.%s.csv", base, tag, value);
		else
			sprintf(myPhiGemmHdl.filename, "%s.%d.csv", base, tag);
	}
#endif

	// Debug output
#if defined(__PHIGEMM_DEBUG)
	printf ("[PHIGEMM_DEBUG] GEMM split : %f \n", myPhiGemmHdl.SPLIT);
	printf ("[PHIGEMM_DEBUG] UPPER_LIMIT_K : %f \n", myPhiGemmHdl.UPPER_LIMIT_K);
	printf ("[PHIGEMM_DEBUG] UPPER_LIMIT_MN : %f \n", myPhiGemmHdl.UPPER_LIMIT_NM);
	printf ("[PHIGEMM_DEBUG] LOWER_LIMIT : %f \n", myPhiGemmHdl.LOWER_LIMIT);
	printf ("[PHIGEMM_DEBUG] SPLITK_GEMM : %f \n", myPhiGemmHdl.PHI_SPLITK_GEMM);
	printf ("[PHIGEMM_DEBUG] SPLITK_FACTOR : %f \n", myPhiGemmHdl.PHI_SPLITK_FACTOR);
#endif

}
