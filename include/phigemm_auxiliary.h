/*
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

#include "phigemm_common.h"

#include "cublas_api.h"
#include "cublas_v2.h"

#define GEMM_ADD(m, n, k) ((m) * (n) * (k))
#define GEMM_MUL(m, n, k) ((m) * (n) * (k))

#ifdef __cplusplus
extern "C"
{
#endif

cudaStream_t  phiStreams[ NSTREAMS * MAX_GPUS ];
cublasHandle_t phiHandles[ NSTREAMS * MAX_GPUS ];

int phiGemmNumDevices;
float phiGemmSplitFactor[4];
int phiGemmCPUThreads;

phiGemmMemDevPtr dev_scratch;
phiGemmMemSizes scratch_size;
phiGemmDeviceIds deviceIds;

void estmSplitFactor(const char* optype, char transa, char transb);

size_t memOccupancy(int is_splitA, float split, int m_in, int n_in, int k_in);

void bestFit(int is_splitA, float split, int m, int n, int k, int type_size, int *p1, int *p2);

double phigemm_cclock(void);

#ifdef __cplusplus
}
#endif

#endif // __PHIGEMM_AUXILIARY_H__
