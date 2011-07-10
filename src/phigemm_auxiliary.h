/*
 * Copyright (C) 2010-2011 Irish Centre for High-End Computing (ICHEC)
 *
 * This file is distributed under the terms of the
 * GNU General Public License. See the file `License'
 * in the root directory of the present distribution,
 * or http://www.gnu.org/copyleft/gpl.txt .
 *
 * author(s):	Philip Yang   (phi@cs.umd.edu)
 * 				Filippo Spiga (filippo.spiga@ichec.ie)
 * 				Ivan Girotto  (ivan.girotto@ichec.ie)
 *
 */

#ifndef __PHIGEMM_AUXILIARY_H__
#define __PHIGEMM_AUXILIARY_H__

#include "phigemm_common.h"

#include "cublas_api.h"
#include "cublas_v2.h"

#define imin(a,b) (((a)<(b))?(a):(b))
#define imax(a,b) (((a)<(b))?(b):(a))

#ifdef __cplusplus
extern "C"
{
#endif

#if (defined __PHIGEMM_DEBUG || defined __PHIGEMM_PROFILE)
typedef struct timestruct
{
	unsigned int sec;
	unsigned int usec;
} TimeStruct;
TimeStruct get_current_time(void);
double GetTimerValue(TimeStruct time_1, TimeStruct time_2);
#endif

void selfPhigemmInit();

void estmSplitFactor(const char* optype, char transa, char transb);

cudaStream_t  phiStreams[ NSTREAM_PER_DEVICE * MAX_GPUS ];
cublasHandle_t phiHandles[ NSTREAM_PER_DEVICE * MAX_GPUS ];
int phiGemmNumDevices;

float phiGemmSplitFactor[3];
phiGemmMemDevPtr dev_scratch;
phiGemmMemSizes scratch_size;
phiGemmDeviceIds deviceIds;

#ifdef __PHIGEMM_DEBUG
double phigemm_cclock(void);
#endif

//extern cudaStream_t phiStreams[MAX_GPUS*NSTREAM_PER_DEVICE];
//extern cublasHandle_t phiHandles[MAX_GPUS*NSTREAM_PER_DEVICE];

#ifdef __cplusplus
}
#endif

#endif // __PHIGEMM_AUXILIARY_H__
