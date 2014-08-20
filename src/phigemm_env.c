/*****************************************************************************\
 * Copyright (C) 2011-2014 Quantum ESPRESSO Foundation
 * Copyright (C) 2010-2011 Irish Centre for High-End Computing (ICHEC)
 *
 * This file is distributed under the terms of the
 * GNU General Public License. See the file `License'
 * in the root directory of the present distribution,
 * or http://www.gnu.org/copyleft/gpl.txt .
 *
 * Filippo Spiga (filippo.spiga@quantum-espresso.org)
\*****************************************************************************/

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
	 * myPhiGemmTng.split                     --> PHI_GEMM_SPLIT
	 * myPhiGemmTng.SPLITK_FACTOR             --> PHI_SPLITK_FACTOR
	 * myPhiGemmTng.THRESHOLD                 --> PHI_THRESHOLD
	 * myPhiGemmTng.SPLITK_GEMM               --> PHI_SPLITK_DGEMM
	 * myPhiGemmTng.LOWER_LIMIT               --> PHI_LOWER_LIMIT
	 * myPhiGemmTng.UPPER_LIMIT_NM            --> PHI_UPPER_LIMIT_NM
	 * myPhiGemmTng.UPPER_LIMIT_K             --> PHI_UPPER_LIMIT_K
	 *
	 * myPhiGemmEnv.filename                  --> base + PHI_PROFILE_PREFIX
	 *
	 */

	float envar;
	char *value = NULL;

	/* SPLIT */
	value = getenv("PHI_GEMM_SPLIT");
	if (value != NULL)
	{
		envar = atof(value);
	} else {
		/* Default if no env variable is specified */
		envar = 1.0;
	}
	myPhiGemmTng.split = envar;
	// free(value); value = NULL;

	/* SPLITK_FACTOR */
	value = getenv("PHI_SPLITK_FACTOR");
	if (value != NULL)
	{
		envar = atof(value);

	} else {
		/* Default if no env variable is specified */
		envar = __SPLITK_FACTOR;
	}
	myPhiGemmTng.SPLITK_FACTOR = envar;

	/* THRESHOLD */
	value = getenv("PHI_THRESHOLD");
	if (value != NULL)
	{
		envar = atof(value);

	} else {
		/* Default if no env variable is specified */
		envar = (int) __SPLITK_FACTOR*1.5;
	}
	myPhiGemmTng.THRESHOLD = envar;

	/* SPLITK_GEMM */
	value = getenv("PHI_SPLITK_GEMM");
	if (value != NULL)
	{
		envar = atoi(value);

	} else {
		/* Default if no env variable is specified */
		envar = __SPLITK_GEMM;
	}
	myPhiGemmTng.SPLITK_GEMM = envar;

	/* LOWER_LIMIT */
	value = getenv("PHI_LOWER_LIMIT");
	if (value != NULL)
	{
		envar = atoi(value);
	} else {
		/* Default if no env variable is specified */
		envar = __LOWER_LIMIT;
	}
	myPhiGemmTng.LOWER_LIMIT = envar;

	/* UPPER_LIMIT_NM */
	value = getenv("PHI_UPPER_LIMIT_NM");
	if (value != NULL)
	{
		envar = atoi(value);
	} else {
		/* Default if no env variable is specified */
		envar = __UPPER_LIMIT_NM;
	}
	myPhiGemmTng.UPPER_LIMIT_NM = envar;

	/* UPPER_LIMIT_K */
	value = getenv("PHI_UPPER_LIMIT_K");
	if (value != NULL)
	{
		envar = atoi(value);
	} else {
		/* Default if no env variable is specified */
		envar = __UPPER_LIMIT_K;
	}
	myPhiGemmTng.UPPER_LIMIT_K = envar;

#if defined(__PHIGEMM_PROFILE)
	/* Create file descriptor where store the profiling information */

	value = getenv("PHI_PROFILE_PREFIX");

	if (tag < 0) {
		if (value != NULL)
			sprintf(myPhiGemmEnv.filename, "%s.%s.csv", base, value);
		else
			sprintf(myPhiGemmEnv.filename, "%s.csv", base);
	} else {
		if (value != NULL)
			sprintf(myPhiGemmEnv.filename, "%s.%d.%s.csv", base, tag, value);
		else
			sprintf(myPhiGemmEnv.filename, "%s.%d.csv", base, tag);
	}
#endif

	// This is a fixed value, not sure how much it impacts varying it...
	myPhiGemmTng.THRESHOLD = myPhiGemmTng.SPLITK_FACTOR*1.5;

#if defined(__PHIGEMM_DEBUG_2)
		printf ("[PHIGEMM_DEBUG][2] GEMM split factor from environment variable: %f \n", myPhiGemmTng.split);
		printf ("[PHIGEMM_DEBUG][2] SPLITK_FACTOR from environment variable: %f \n", myPhiGemmTng.SPLITK_FACTOR);
		printf ("[PHIGEMM_DEBUG][2] THRESHOLD from environment variable: %f \n", myPhiGemmTng.THRESHOLD);
		printf ("[PHIGEMM_DEBUG][2] SPLITK_GEMM from environment variable: %d \n", myPhiGemmTng.SPLITK_GEMM);
		printf ("[PHIGEMM_DEBUG][2] UPPER_LIMIT_MN from environment variable: %d \n", myPhiGemmTng.UPPER_LIMIT_NM);
		printf ("[PHIGEMM_DEBUG][2] UPPER_LIMIT_K from environment variable: %d \n", myPhiGemmTng.UPPER_LIMIT_K);
		printf ("[PHIGEMM_DEBUG][2] LOWER_LIMIT from environment variable: %d \n", myPhiGemmTng.LOWER_LIMIT);
		fflush(stdout);
#endif
}
