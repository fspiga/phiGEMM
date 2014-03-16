/*
 * Copyright (C) 2011-2012 Quantum ESPRESSO Foundation
 * Copyright (C) 2010-2011 Irish Centre for High-End Computing (ICHEC)
 * Copyright (C) 2001-2011 Quantum ESPRESSO group
 *
 * This file is distributed under the terms of the
 * GNU General Public License. See the file `License'
 * in the root directory of the present distribution,
 * or http://www.gnu.org/copyleft/gpl.txt .
 *
 */

#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime.h>
//#include "cublas_api.h"
//#include "cublas_v2.h"

#ifndef __PHIGEMM_CUDA_TEST_H

#define __PHIGEMM_CUDA_TEST_H

#define __SCALING_MEM_FACTOR__ 0.95
#define MAX_GPUS 4

typedef void* phiTestCudaMemDevPtr[MAX_GPUS];
typedef size_t phiTestCudaMemSizes[MAX_GPUS];
typedef int phiTestCudaDevicesBond[MAX_GPUS];

extern phiTestCudaMemDevPtr dev_scratch_test;
extern phiTestCudaMemSizes cuda_memory_allocated;
extern phiTestCudaDevicesBond gpu_bonded;

#endif // __PHIGEMM_CUDA_TEST_H
