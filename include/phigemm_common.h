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

#ifndef __PHIGEMM_COMMON_H__
#define __PHIGEMM_COMMON_H__

#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <dlfcn.h>
#include <ctype.h>

#include "cublas_v2.h"

#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>

// phiGEMM data-type <--> CUDA data-type
typedef cuComplex phiComplex;
typedef cuDoubleComplex phiDoubleComplex;

#define phigemm_get_real_part(data) data.x
#define phigemm_get_img_part(data) data.y
#define phigemm_set_real_part(data, value) data.x = value
#define phigemm_set_img_part(data, value) data.y = value

/* --------------------------- MAIN DEFAULF VALUES ------------------------- */

// This feature is not tested/checked since long time...
#if defined (__PHIGEMM_MULTI_STREAMS)
#define NSTREAMS 2
#else
#define NSTREAMS 1
#endif

#ifndef __SCALING_INIT_MEM
#define __SCALING_INIT_MEM 0.95
#endif

#if defined(__CUDA_GET_MEM_HACK)
#ifndef __GPU_MEM_AMOUNT_HACK__
#define __GPU_MEM_AMOUNT_HACK__ 2400000000
#endif
#endif

#ifndef __SPLITK_FACTOR
#define __SPLITK_FACTOR 20
#endif

#ifndef __SPLITK_GEMM
#define __SPLITK_GEMM 2048
#endif

#ifndef __LOWER_LIMIT
#if defined(__PHIGEMM_ENABLE_SPECIALK)
#define __LOWER_LIMIT 63
#else
#define __LOWER_LIMIT 127
#endif
#endif

#ifndef __UPPER_LIMIT_NM
#define __UPPER_LIMIT_NM 255
#endif

#ifndef __UPPER_LIMIT_K
#define __UPPER_LIMIT_K 1023
#endif

#if defined(__PHIGEMM_PINNED) || defined(__PHIGEMM_MULTI_GPU)
#define __PHIGEMM_EVENTS 6
#else
#define __PHIGEMM_EVENTS 7
#endif

/* -------------------------------- TYPEDEFS ------------------------------- */

typedef void* phiGemmMemDevPtr[NSTREAMS];

typedef size_t phiGemmMemSizes[NSTREAMS];

typedef int phiGemmDeviceIds[NSTREAMS];

typedef struct phiGemmEnv
{
#if defined(__PHIGEMM_PROFILE)
	FILE *profileFile;
	char filename [ FILENAME_MAX ];
#endif
} phiGemmEnv_t;

typedef struct phiGemmHandler
{
	phiGemmMemDevPtr pmem;
	phiGemmMemSizes smem;
	phiGemmDeviceIds devId;
	cudaStream_t  stream[ NSTREAMS];
	cublasHandle_t handle[ NSTREAMS ];
} phiGemmHandler_t;


typedef struct phiGemmTuning
{
	float split;
	float SPLITK_FACTOR;
	int SPLITK_GEMM;
	int LOWER_LIMIT;
	int UPPER_LIMIT_NM;
	int UPPER_LIMIT_K;
} phiGemmTuning_t;


/* ------------------------------ OTHER MACROS ----------------------------- */

#define EVENIZE(x) ( ((x)%2==0) ? (x) : (x)+1 )

#define GEMM_ADD(m, n, k) ((m) * (n) * (k))
#define GEMM_MUL(m, n, k) ((m) * (n) * (k))

#define imin(a,b) (((a)<(b))?(a):(b))
#define imax(a,b) (((a)<(b))?(b):(a))

#endif // __PHIGEMM_COMMON_H__
