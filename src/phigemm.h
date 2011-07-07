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

#ifndef __PHIGEMM_H__
#define __PHIGEMM_H__

#include "phigemm_common.h"

#ifdef __cplusplus
extern "C"
{
#endif

/* C interface */

void phiGemmInit( int nGPU, phiGemmMemDevPtr* dev_ptr, phiGemmMemSizes* dev_memsize, int * deviceToBond);

void phiGemmShutdown();

int phiGemmIsInit();

int phiGemmGetRank();

void phigemmSetSplitFactor(float *x);

#if defined __PHIGEMM_PROFILE
void phiSgemm (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const float *alpha,
		const float *A, const int *lda, const float *B,
		const int *ldb, const float *beta, float *C, const int *ldc,
		const char *file, const int line );

void phiDgemm (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const double *alpha,
		const double *A, const int *lda, const double *B,
		const int *ldb, const double *beta, double *C, const int *ldc,
		const char *file, const int line );

void phiZgemm (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const cuDoubleComplex *alpha,
		const cuDoubleComplex *A, const int *lda, const cuDoubleComplex *B,
		const int *ldb, const cuDoubleComplex *beta, cuDoubleComplex *C,
		const int *ldc, const char *file, const int line );
#else
void phiSgemm (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const float *alpha,
		const float *A, const int *lda, const float *B,
		const int *ldb, const float *beta, float *C, const int *ldc);

void phiDgemm (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const double *alpha,
		const double *A, const int *lda, const double *B,
		const int *ldb, const double *beta, double *C, const int *ldc);

void phiZgemm (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const cuDoubleComplex *alpha,
		const cuDoubleComplex *A, const int *lda, const cuDoubleComplex *B,
		const int *ldb, const cuDoubleComplex *beta, cuDoubleComplex *C,
		const int *ldc);
#endif

/* Fortran interface */

void phigemminit_(int nGPU, phiGemmMemDevPtr* dev_ptr, phiGemmMemSizes* dev_memsize, int * deviceToBond);

void phigemmshutdown_();

int phigemmisinit_();

int phigemmgetrank_();

void phigemmsetsplitfactor_(float *x);

#if defined __PHIGEMM_PROFILE
void phisgemm_ (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const float *alpha,
		const float *A, const int *lda, const float *B,
		const int *ldb, const float *beta, float *C, const int *ldc,
		const char *file, const int line );

void phidgemm_ (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const double *alpha,
		const double *A, const int *lda, const double *B,
		const int *ldb, const double *beta, double *C, const int *ldc,
		const char *file, const int line );

void phizgemm_ (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const cuDoubleComplex *alpha,
		const cuDoubleComplex *A, const int *lda, const cuDoubleComplex *B,
		const int *ldb, const cuDoubleComplex *beta, cuDoubleComplex *C,
		const int *ldc, const char *file, const int line );
#else
void phisgemm_ (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const float *alpha,
		const float *A, const int *lda, const float *B,
		const int *ldb, const float *beta, float *C, const int *ldc);

void phidgemm_ (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const double *alpha,
		const double *A, const int *lda, const double *B,
		const int *ldb, const double *beta, double *C, const int *ldc);

void phizgemm_ (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const cuDoubleComplex *alpha,
		const cuDoubleComplex *A, const int *lda, const cuDoubleComplex *B,
		const int *ldb, const cuDoubleComplex *beta, cuDoubleComplex *C,
		const int *ldc);
#endif

#ifdef __cplusplus
}
#endif

#endif //__PHIGEMM_H__
