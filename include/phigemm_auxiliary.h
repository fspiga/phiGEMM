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

#ifndef __PHIGEMM_AUXILIARY_H__
#define __PHIGEMM_AUXILIARY_H__

#include <stdlib.h>
#include <stdio.h>

#if !defined(__PHIGEMM_CPUONLY)
#include "cuda.h"
#include "cublas_api.h"
#include <cuda_runtime.h>
#include "cublas_v2.h"
#endif

#include "phigemm_common.h"

#ifdef __cplusplus
extern "C"
{
#endif

/* ------------------------- SHARED DATA STRUCTURES ------------------------ */

extern phiGemmEnv_t myPhiGemmEnv;

#if !defined(__PHIGEMM_CPUONLY)
extern phiGemmHandler_t myPhiGemmHdl;
#endif

extern phiGemmTuning_t myPhiGemmTng;

/* ------------------------------------------------------------------------- */


/* --------------------- INTERNAL FUNCTIONS PROTOTYPES --------------------- */

#if !defined(__PHIGEMM_CPUONLY)
int phiGemmIsInternalMemAlloc();

int phiGemmIsExternalMemAlloc();

void estmSplitFactor(const char* optype, char transa, char transb);

size_t memOccupancy(int is_splitA, float split, int m_in, int n_in, int k_in);

void bestFit(int is_splitA, float split, int m, int n, int k, int type_size, int *p1, int *p2);

int cpuGPUheuristic(int m, int n, int k, char type);

void phiGemmInitScratchMemory( );
#endif

double phigemm_cclock(void);

/* ------------------------------------------------------------------------- */

#ifdef __cplusplus
}
#endif

#endif // __PHIGEMM_AUXILIARY_H__
