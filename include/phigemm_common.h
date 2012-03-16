/*
 * Copyright (C) 2010-2011 Irish Centre for High-End Computing (ICHEC)
 *
 * This file is distributed under the terms of the
 * GNU General Public License. See the file `License'
 * in the root directory of the present distribution,
 * or http://www.gnu.org/copyleft/gpl.txt .
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

#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>

#if defined (__PHIGEMM_MULTI_STREAMS)
#define MAX_GPUS 8
#define NSTREAMS 2
#else
#define MAX_GPUS 8
#define NSTREAMS 1
#endif

#define PHIGEMM_SPLITK_DGEMM 1024
#define PHIGEMM_SPLITK_ZGEMM 1024

#define SPLITK_FACTOR 20

typedef void* phiGemmMemDevPtr[MAX_GPUS*NSTREAMS];
typedef size_t phiGemmMemSizes[MAX_GPUS*NSTREAMS];
typedef int phiGemmDeviceIds[MAX_GPUS*NSTREAMS];

#define imin(a,b) (((a)<(b))?(a):(b))
#define imax(a,b) (((a)<(b))?(b):(a))

#endif // __PHIGEMM_COMMON_H__
