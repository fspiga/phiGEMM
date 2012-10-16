/*
 * Copyright (C) 2011-2012 Quantum ESPRESSO Foundation
 * Copyright (C) 2010-2011 Irish Centre for High-End Computing (ICHEC)
 *
 * This file is distributed under the terms of the
 * GNU General Public License. See the file `License'
 * in the root directory of the present distribution,
 * or http://www.gnu.org/copyleft/gpl.txt .
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
	 * PHIGEMM_TUNE         = Apply auto-tune strategy (default: 1)
	 * PHIGEMM_SPLITK       = (default: 1)
	 * PHIGEMM_LOG_VERBOS   = (default: 0, dependency: __PHIGEMM_PROFILING)
	 *
	 * PHIGEMM_SGEMM_SPLIT  = (default: )
	 * PHIGEMM_CGEMM_SPLIT  = (default: )
	 * PHIGEMM_DGEMM_SPLIT  = (default: )
	 * PHIGEMM_ZGEMM_SPLIT  = (default: )
	 *
	 * PHIGEMM_LOWER_NM     = (default: )
	 * PHIGEMM_UPPER_NM     = (default: )
	 * PHIGEMM_UPPER_K      = (default: )
	 * PHIGEMM_SPLITK_MN    = (default: 20)
	 *
	 * PHIGEMM_SPLITK_D     = (default: 2048)
	 * PHIGEMM_SPLITK_Z     = (default: 2048)
	 *
	 * PHIGEMM_TUNE_BAL_L   = negative threshold limit around perfect balance (default: )
	 * PHIGEMM_TUNE_BAL_P   = positive threshold limit around perfect balance (default: )
	 * PHIGEMM_TUNE_SHIFT_P = positive split shift (default: )
	 * PHIGEMM_TUNE_SHIFT_L = positive split shift (default: )
	 *
	 *
	 * int phiGemmControl[4]={PHIGEMM_TUNE, PHIGEMM_SPLITK, PHIGEMM_LOG_VERBOS}
	 *
	 * float phiGemmSplitFactor[4]={PHIGEMM_SGEMM_SPLIT, PHIGEMM_CGEMM_SPLIT,
	 * 								PHIGEMM_DGEMM_SPLIT, PHIGEMM_ZGEMM_SPLIT}
	 *
	 * int phiGemmSpecialK[4]={PHIGEMM_SPLITK, PHIGEMM_LOWER_NM, PHIGEMM_UPPER_NM,
	 * 							PHIGEMM_UPPER_K }
	 *
	 * int phiGemmKDimBlocks[4]={ *not used*, *not used*,
	 * 									PHIGEMM_SPLITK_D, PHIGEMM_SPLITK_Z}
	 *
	 * float phiGemmAutoTune[4]={PHIGEMM_TUNE_BAL_L, PHIGEMM_TUNE_BAL_P,
	 * 								PHIGEMM_TUNE_SHIFT_P, PHIGEMM_TUNE_SHIFT_L}
	 *
	 */

	float envar_split;
	char *value = NULL;

	/* split factor may vary between S/D/Z GEMMs */

	/* SGEMM */
	value = getenv("PHI_SGEMM_SPLIT");
	if (value != NULL)
	{
		envar_split = atof(value);
#if defined(__PHIGEMM_DEBUG)
		printf ("[PHIGEMM_DEBUG] SGEMM split factor from environment variable: %f \n", envar_split);
#endif
	} else {
		/* Default split if no env variables are specified */
		envar_split = 0.85;
#if defined(__PHIGEMM_DEBUG)
		printf ("[PHIGEMM_DEBUG] SGEMM default split factor: %f \n", envar_split);
#endif
	}
	myPhiGemmTng.split[0] = envar_split;
	myPhiGemmTng.prevSplit[0] = envar_split;
	myPhiGemmTng.lpSplit[0] = 0.995 ;

	/* DGEMM */
	value = getenv("PHI_DGEMM_SPLIT");
	if (value != NULL)
	{
		envar_split = atof(value);
#if defined(__PHIGEMM_DEBUG)
		printf ("[PHIGEMM_DEBUG] DGEMM split factor from environment variable: %f \n", envar_split);
#endif
	} else {
		/* Default split if no env variables are specified */
		envar_split = 0.875;
#if defined(__PHIGEMM_DEBUG)
		printf ("[PHIGEMM_DEBUG] DGEMM default split factor: %f \n", envar_split);
#endif
	}
	myPhiGemmTng.split[1] = envar_split;
	myPhiGemmTng.prevSplit[1] = envar_split;
	myPhiGemmTng.lpSplit[1] = 0.995 ;

	/* CGEMM */
	value = getenv("PHI_CGEMM_SPLIT");
	if (value != NULL)
	{
		envar_split = atof(value);
#if defined(__PHIGEMM_DEBUG)
		printf ("[PHIGEMM_DEBUG] CGEMM split factor from environment variable: %f \n", envar_split);
#endif
	} else {

		/* Default split if no env variables are specified */
		envar_split = 0.9;
#if defined(__PHIGEMM_DEBUG)
		printf ("[PHIGEMM_DEBUG] CGEMM  default split factor: %f \n", envar_split);
#endif
	}
	myPhiGemmTng.split[2] = envar_split;
	myPhiGemmTng.prevSplit[2] = envar_split;
	myPhiGemmTng.lpSplit[2] = 0.995 ;

	/* ZGEMM */
	value = getenv("PHI_ZGEMM_SPLIT");
	if (value != NULL)
	{
		envar_split = atof(value);
#if defined(__PHIGEMM_DEBUG)
		printf ("[PHIGEMM_DEBUG] ZGEMM split factor from environment variable: %f \n", envar_split);
#endif
	} else {

		/* Default split if no env variables are specified */
		envar_split = 0.925;
#if defined(__PHIGEMM_DEBUG)
		printf ("[PHIGEMM_DEBUG] ZGEMM  default split factor: %f \n", envar_split);
#endif
	}
	myPhiGemmTng.split[3] = envar_split;
	myPhiGemmTng.prevSplit[3] = envar_split;
	myPhiGemmTng.lpSplit[3] = 0.995 ;

	/* This is to avoid not-defined OMP_NUM_THREADS in the environment.
	 * Default threads num = 1 */
	value = getenv("OMP_NUM_THREADS");
	if (value != NULL)
	{
		myPhiGemmEnv.cores = atoi(value);
	} else {

		/* Default threads num = 1 */
		myPhiGemmEnv.cores = 1;
	}
#if defined(__PHIGEMM_DEBUG)
	printf ("[PHIGEMM_DEBUG] phiGemmCPUThreads: %d \n", myPhiGemmEnv.cores);
#endif

}
