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


#if 0
// Replacements if CUDA native data-types are not available
typedef float phiComplex[2];
typedef double phiDoubleComplex[2];

#define phigemm_get_real_part(data) data[0]
#define phigemm_get_img_part(data) data[1]
#define phigemm_set_real_part(data, value) data[0] = value
#define phigemm_set_img_part(data, value) data[1] = value

#endif

/* --------------------------- MAIN DEFAULF VALUES ------------------------- */

#define MAX_GPUS 4

// This feature is not tested/checked since long time...
#if defined (__PHIGEMM_MULTI_STREAMS)
#define NSTREAMS 2
#else
#define NSTREAMS 1
#endif

#define __SCALING_INIT_MEM 0.95
#define __SPLITK_FACTOR 20
#define __SPLITK_GEMM 2048
#define __LOWER_LIMIT 63
//#define __LOWER_LIMIT 127
#define __UPPER_LIMIT_NM 255
#define __UPPER_LIMIT_K 1023

#if defined(__PHIGEMM_PINNED) || defined(__PHIGEMM_MULTI_GPU)
#define __PHIGEMM_EVENTS 6
#else
#define __PHIGEMM_EVENTS 7
#endif

/* -------------------------------- TYPEDEFS ------------------------------- */

typedef void* phiGemmMemDevPtr[MAX_GPUS*NSTREAMS];

typedef size_t phiGemmMemSizes[MAX_GPUS*NSTREAMS];

typedef int phiGemmDeviceIds[MAX_GPUS*NSTREAMS];

typedef struct phiGemmEnv
{
	int numDevices;
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
	cudaStream_t  stream[ NSTREAMS * MAX_GPUS ];
	cublasHandle_t handle[ NSTREAMS * MAX_GPUS ];
} phiGemmHandler_t;

typedef struct phiGemmTuning
{
	float split;
	float SPLITK_FACTOR;
	float THRESHOLD;
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
