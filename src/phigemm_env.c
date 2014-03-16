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


/*
 * Name			: readEnv
 * Description	: the method read from the environment some special variables
 * 				  and eventually overwrite the defaults
 * Visibility	: this file only
 */
void readEnv()
{
	/*
	 * <phiGEMM data structure>.<field>       --> env variable
	 *
	 * myPhiGemmTng.split[0] = prevSplit[0]   --> PHI_SGEMM_SPLIT -- DEPRECATED
	 * myPhiGemmTng.lpSplit[0]                --> 0.995
	 * myPhiGemmTng.split[1] = prevSplit[1]   --> PHI_DGEMM_SPLIT
	 * myPhiGemmTng.lpSplit[1]                --> 0.995
	 * myPhiGemmTng.split[2] = prevSplit[2]   --> PHI_CGEMM_SPLIT -- DEPRECATED
	 * myPhiGemmTng.lpSplit[2]                --> 0.995
	 * myPhiGemmTng.split[3] = prevSplit[3]   --> PHI_ZGEMM_SPLIT
	 * myPhiGemmTng.lpSplit[3]                --> 0.995
	 * myPhiGemmTng.SPLITK_FACTOR             --> PHI_SPLITK_FACTOR
	 * myPhiGemmTng.THRESHOLD                 --> PHI_THRESHOLD
	 * myPhiGemmTng.SPLITK_DGEMM              --> PHI_SPLITK_DGEMM
	 * myPhiGemmTng.SPLITK_ZGEMM              --> PHI_SPLITK_ZGEMM
	 * myPhiGemmTng.LOWER_LIMIT               --> PHI_LOWER_LIMIT
	 * myPhiGemmTng.UPPER_LIMIT_NM            --> PHI_UPPER_LIMIT_NM
	 * myPhiGemmTng.UPPER_LIMIT_K             --> PHI_UPPER_LIMIT_K
	 *
	 * myPhiGemmEnv.cores                     --> OMP_NUM_THREADS
	 */

	float envar;
	char *value = NULL;

#if !defined(__PHIGEMM_CPUONLY)
	/* SGEMM -- DEPRECATED */
	myPhiGemmTng.split[0] = 1.0;
	myPhiGemmTng.prevSplit[0] = 1.0;
	myPhiGemmTng.lpSplit[0] = 1.0 ;

	/* DGEMM */
	value = getenv("PHI_DGEMM_SPLIT");
	if (value != NULL)
	{
		envar = atof(value);
#if defined(__PHIGEMM_DEBUG)
		printf ("[PHIGEMM_DEBUG] DGEMM split factor from environment variable: %f \n", envar);
#endif
	} else {
		/* Default if no env variable is specified */
		envar = 0.95;
#if defined(__PHIGEMM_DEBUG)
		printf ("[PHIGEMM_DEBUG] DGEMM default split factor: %f \n", envar);
#endif
	}
	myPhiGemmTng.split[1] = envar;
	myPhiGemmTng.prevSplit[1] = envar;
	myPhiGemmTng.lpSplit[1] = 0.995 ;

	/* CGEMM -- DEPRECATED */
	myPhiGemmTng.split[2] = 1.0;
	myPhiGemmTng.prevSplit[2] = 1.0;
	myPhiGemmTng.lpSplit[2] = 1.0 ;

	/* ZGEMM */
	value = getenv("PHI_ZGEMM_SPLIT");
	if (value != NULL)
	{
		envar = atof(value);
#if defined(__PHIGEMM_DEBUG)
		printf ("[PHIGEMM_DEBUG] ZGEMM split factor from environment variable: %f \n", envar);
#endif
	} else {
		/* Default if no env variable is specified */
		envar = 0.95;
#if defined(__PHIGEMM_DEBUG)
		printf ("[PHIGEMM_DEBUG] ZGEMM  default split factor: %f \n", envar);
#endif
	}
	myPhiGemmTng.split[3] = envar;
	myPhiGemmTng.prevSplit[3] = envar;
	myPhiGemmTng.lpSplit[3] = 0.995 ;

	/* SPLITK_FACTOR */
	value = getenv("PHI_SPLITK_FACTOR");
	if (value != NULL)
	{
		envar = atof(value);
#if defined(__PHIGEMM_DEBUG)
		printf ("[PHIGEMM_DEBUG] SPLITK_FACTOR from environment variable: %f \n", envar);
#endif
	} else {
		/* Default if no env variable is specified */
		envar = __SPLITK_FACTOR;
#if defined(__PHIGEMM_DEBUG)
		printf ("[PHIGEMM_DEBUG] SPLITK_FACTOR default: %f \n", envar);
#endif
	}
	myPhiGemmTng.SPLITK_FACTOR = envar;


	/* THRESHOLD */
	value = getenv("PHI_THRESHOLD");
	if (value != NULL)
	{
		envar = atof(value);
#if defined(__PHIGEMM_DEBUG)
		printf ("[PHIGEMM_DEBUG] THRESHOLD from environment variable: %f \n", envar);
#endif
	} else {
		/* Default if no env variable is specified */
		envar = (int) __SPLITK_FACTOR*1.5;
#if defined(__PHIGEMM_DEBUG)
		printf ("[PHIGEMM_DEBUG] THRESHOLD default: %f \n", envar);
#endif
	}
	myPhiGemmTng.THRESHOLD = envar;


	/* SPLITK_DGEMM */
	value = getenv("PHI_SPLITK_DGEMM");
	if (value != NULL)
	{
		envar = atoi(value);
#if defined(__PHIGEMM_DEBUG)
		printf ("[PHIGEMM_DEBUG] SPLITK_DGEMM from environment variable: %f \n", envar);
#endif
	} else {
		/* Default if no env variable is specified */
		envar = __SPLITK_DGEMM;
#if defined(__PHIGEMM_DEBUG)
		printf ("[PHIGEMM_DEBUG] SPLITK_DGEMM default: %f \n", envar);
#endif
	}
	myPhiGemmTng.SPLITK_DGEMM = envar;


	/* SPLITK_ZGEMM */
	value = getenv("PHI_SPLITK_ZGEMM");
	if (value != NULL)
	{
		envar = atoi(value);
#if defined(__PHIGEMM_DEBUG)
		printf ("[PHIGEMM_DEBUG] SPLITK_ZGEMM from environment variable: %f \n", envar);
#endif
	} else {
		/* Default if no env variable is specified */
		envar = __SPLITK_ZGEMM;
#if defined(__PHIGEMM_DEBUG)
		printf ("[PHIGEMM_DEBUG] SPLITK_ZGEMM default: %f \n", envar);
#endif
	}
	myPhiGemmTng.SPLITK_ZGEMM = envar;


	/* LOWER_LIMIT */
	value = getenv("PHI_LOWER_LIMIT");
	if (value != NULL)
	{
		envar = atoi(value);
#if defined(__PHIGEMM_DEBUG)
		printf ("[PHIGEMM_DEBUG] LOWER_LIMIT from environment variable: %f \n", envar);
#endif
	} else {
		/* Default if no env variable is specified */
		envar = __LOWER_LIMIT;
#if defined(__PHIGEMM_DEBUG)
		printf ("[PHIGEMM_DEBUG] LOWER_LIMIT default: %f \n", envar);
#endif
	}
	myPhiGemmTng.LOWER_LIMIT = envar;


	/* UPPER_LIMIT_NM */
	value = getenv("PHI_UPPER_LIMIT_NM");
	if (value != NULL)
	{
		envar = atoi(value);
#if defined(__PHIGEMM_DEBUG)
		printf ("[PHIGEMM_DEBUG] UPPER_LIMIT_K from environment variable: %f \n", envar);
#endif
	} else {
		/* Default if no env variable is specified */
		envar = __UPPER_LIMIT_NM;
#if defined(__PHIGEMM_DEBUG)
		printf ("[PHIGEMM_DEBUG] UPPER_LIMIT_K default: %f \n", envar);
#endif
	}
	myPhiGemmTng.UPPER_LIMIT_NM = envar;


	/* UPPER_LIMIT_K */
	value = getenv("PHI_UPPER_LIMIT_K");
	if (value != NULL)
	{
		envar = atoi(value);
#if defined(__PHIGEMM_DEBUG)
		printf ("[PHIGEMM_DEBUG] UPPER_LIMIT_K from environment variable: %f \n", envar);
#endif
	} else {
		/* Default if no env variable is specified */
		envar = __UPPER_LIMIT_K;
#if defined(__PHIGEMM_DEBUG)
		printf ("[PHIGEMM_DEBUG] UPPER_LIMIT_K default: %f \n", envar);
#endif
	}
	myPhiGemmTng.UPPER_LIMIT_K = envar;

#endif

	/* This is to avoid not-defined OMP_NUM_THREADS in the environment.
	 * Default threads num = 1 */
	value = getenv("OMP_NUM_THREADS");
	if (value != NULL)
	{
		myPhiGemmEnv.cores = atoi(value);
#if defined(__PHIGEMM_DEBUG)
		printf ("[PHIGEMM_DEBUG] OMP_NUM_THREADS from environment variable: %d \n", myPhiGemmEnv.cores);
#endif
	} else {
		/* Default threads num = 1 */
		myPhiGemmEnv.cores = 1;
#if defined(__PHIGEMM_DEBUG)
		printf ("[PHIGEMM_DEBUG] OMP_NUM_THREADS default (no-threading): %d \n", myPhiGemmEnv.cores);
#endif
	}


}
