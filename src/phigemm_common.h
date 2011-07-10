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

#ifndef __PHIGEMM_COMMON_H__
#define __PHIGEMM_COMMON_H__

#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <dlfcn.h>
#include <ctype.h>

#ifdef __PHIGEMM_PARA
#include <mpi.h>
#endif

#include "cublas_api.h"

#if (defined __PHIGEMM_DEBUG || defined __PHIGEMM_PROFILE)
#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#endif

#if defined (__PHIGEMM_MULTI_STREAMS) && defined(__PHIGEMM_MEM_ASYNC)
#define MAX_GPUS 8
#define NSTREAM_PER_DEVICE 2
#else
#define MAX_GPUS 8
#define NSTREAM_PER_DEVICE 1
#endif

typedef void* phiGemmMemDevPtr[MAX_GPUS*NSTREAM_PER_DEVICE];
typedef size_t phiGemmMemSizes[MAX_GPUS*NSTREAM_PER_DEVICE];
typedef int phiGemmDeviceIds[MAX_GPUS*NSTREAM_PER_DEVICE];

#define imin(a,b) (((a)<(b))?(a):(b))
#define imax(a,b) (((a)<(b))?(b):(a))

#define OnEGiG 1.0737e9

#endif // __PHIGEMM_COMMON_H__
