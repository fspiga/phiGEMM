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

#ifndef __PHIGEMM_AUXILIARY_H__
#define __PHIGEMM_AUXILIARY_H__

#include <stdlib.h>
#include <stdio.h>

#include "cuda.h"
#include "cublas_api.h"
#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "phigemm_common.h"

/* ---------------------------------- MACROS ------------------------------- */

#define __SCALING_MEM_FACTOR__ 0.95

#if defined(__CUDA_GET_MEM_HACK)
#define __GPU_MEM_AMOUNT_HACK__ 2400000000
#endif

#define GEMM_ADD(m, n, k) ((m) * (n) * (k))
#define GEMM_MUL(m, n, k) ((m) * (n) * (k))

//#if defined(PRECISION_D) || defined(PRECISION_S)
//#define PHIGEMM_FLOPS(m, n, k) (      GEMM_MUL(m, n, k) +      GEMM_ADD(m, n, k))
//#else
//#define PHIGEMM_FLOPS(m, n, k) (  6 * GEMM_MUL(m, n, k) +  2 * GEMM_ADD(m, n, k))
//#endif

#if defined(__PHIGEMM_PINNED) || defined(__PHIGEMM_MULTI_GPU)
#define PHIGEMM_EVENTS 6
#else
#define PHIGEMM_EVENTS 7
#endif

/* ------------------------------------------------------------------------- */

#ifdef __cplusplus
extern "C"
{
#endif

/* ------------------------- SHARED DATA STRUCTURES ------------------------ */
/* For AutoTuning purposes*/
float phiGemmPrevSplitFactor[4];
float phiGemmLowerPositiveSplitFactor[4];

/* Control variables (see 'readEnv' in phigemm_auxiliary.c)*/
float phiGemmSplitFactor[4];
float phiGemmAutoTune[4];
int phiGemmSpecialK[4];
int phiGemmKDimBlocks[4];
int phiGemmControl[4];

int phiGemmNumDevices;
int phiGemmCPUThreads;

cudaStream_t  phiStreams[ NSTREAMS * MAX_GPUS ];
cublasHandle_t phiHandles[ NSTREAMS * MAX_GPUS ];

phiGemmMemDevPtr dev_scratch;
phiGemmMemSizes scratch_size;
phiGemmDeviceIds deviceIds;

#if defined(__PHIGEMM_PROFILE)
FILE *phiProfileFile;
#endif
/* ------------------------------------------------------------------------- */

/* --------------------- INTERNAL FUNCTIONS PROTOTYPES --------------------- */

int phiGemmIsInternalMemAlloc();

int phiGemmIsExternalMemAlloc();

void estmSplitFactor(const char* optype, char transa, char transb);

size_t memOccupancy(int is_splitA, float split, int m_in, int n_in, int k_in);

void bestFit(int is_splitA, float split, int m, int n, int k, int type_size, int *p1, int *p2);

int cpuGPUheuristic(int m, int n, int k, char type);

void phiGemmInitScratchMemory( );

double phigemm_cclock(void);

/* ------------------------------------------------------------------------- */

#ifdef __cplusplus
}
#endif

#endif // __PHIGEMM_AUXILIARY_H__
