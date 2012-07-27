/*
 * Copyright (C) 2010-2011 Irish Centre for High-End Computing (ICHEC)
 * Copyright (C) 2011-2012 Quantum ESPRESSO Foundation
 *
 * This file is distributed under the terms of the
 * GNU General Public License. See the file `License'
 * in the root directory of the present distribution,
 * or http://www.gnu.org/copyleft/gpl.txt .
 *
 */

#ifndef __PHIGEMM_H__
#define __PHIGEMM_H__

#include "phigemm_common.h"

#ifdef __cplusplus
extern "C"
{
#endif

/* C interface - does it work? */

void phiGemmInit( int nGPU, phiGemmMemDevPtr* dev_ptr, phiGemmMemSizes* dev_memsize, int * deviceToBond, int tag);

void phiGemmShutdown();

int phiGemmIsInit();

void phigemmSetSplitFactor(float *x);

float phigemmGetSplitFactor(int selection);

void phiGemmSetAvaiableScratchSpace(int gpu_id, size_t new_dev_memsize);


#if defined __PHIGEMM_PROFILE
void phiSgemm (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const float *alpha,
		const float *A, const int *lda, const float *B,
		const int *ldb, const float *beta, float *C, const int *ldc,
		const char *file, const char * line );

void phiDgemm (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const double *alpha,
		const double *A, const int *lda, const double *B,
		const int *ldb, const double *beta, double *C, const int *ldc,
		const char *file, const char * line );

void phidgemm_specialK(const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const double *alpha,
		const double *A, const int *lda, const double *B,
		const int *ldb, const double *beta, double *C, const int *ldc,
		const char *file, const char * line );

void phiCgemm (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const cuComplex *alpha,
		const cuComplex *A, const int *lda, const cuComplex *B,
		const int *ldb, const cuComplex *beta, cuComplex *C,
		const int *ldc, const char *file, const char * line );

void phiZgemm (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const cuDoubleComplex *alpha,
		const cuDoubleComplex *A, const int *lda, const cuDoubleComplex *B,
		const int *ldb, const cuDoubleComplex *beta, cuDoubleComplex *C,
		const int *ldc, const char *file, const char * line );

void phizgemm_specialK (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const cuDoubleComplex *alpha,
		const cuDoubleComplex *A, const int *lda, const cuDoubleComplex *B,
		const int *ldb, const cuDoubleComplex *beta, cuDoubleComplex *C,
		const int *ldc, const char *file, const char * line );
#else
	void phiSgemm (const char *transa, const char *transb, const int *m,
			const int *n, const int *k, const float *alpha,
			const float *A, const int *lda, const float *B,
			const int *ldb, const float *beta, float *C, const int *ldc);

	void phiDgemm (const char *transa, const char *transb, const int *m,
			const int *n, const int *k, const double *alpha,
			const double *A, const int *lda, const double *B,
			const int *ldb, const double *beta, double *C, const int *ldc);

	void phidgemm_specialK(const char *transa, const char *transb, const int *m,
			const int *n, const int *k, const double *alpha,
			const double *A, const int *lda, const double *B,
			const int *ldb, const double *beta, double *C, const int *ldc);

	void phiCgemm (const char *transa, const char *transb, const int *m,
			const int *n, const int *k, const cuComplex *alpha,
			const cuComplex *A, const int *lda, const cuComplex *B,
			const int *ldb, const cuComplex *beta, cuComplex *C,
			const int *ldc);

	void phiZgemm (const char *transa, const char *transb, const int *m,
			const int *n, const int *k, const cuDoubleComplex *alpha,
			const cuDoubleComplex *A, const int *lda, const cuDoubleComplex *B,
			const int *ldb, const cuDoubleComplex *beta, cuDoubleComplex *C,
			const int *ldc);

	void phizgemm_specialK (const char *transa, const char *transb, const int *m,
			const int *n, const int *k, const cuDoubleComplex *alpha,
			const cuDoubleComplex *A, const int *lda, const cuDoubleComplex *B,
			const int *ldb, const cuDoubleComplex *beta, cuDoubleComplex *C,
			const int *ldc);
#endif

/* Fortran interface */

void phigemminit_(int nGPU, phiGemmMemDevPtr* dev_ptr, phiGemmMemSizes* dev_memsize, int * deviceToBond, int tag);

void phigemmshutdown_();

int phigemmisinit_();

void phigemmsetsplitfactor_(float *x);

void phiremmsetavaiablescratchspace_(int gpu_id, size_t new_dev_memsize);

#if defined __PHIGEMM_PROFILE
void phisgemm_ (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const float *alpha,
		const float *A, const int *lda, const float *B,
		const int *ldb, const float *beta, float *C, const int *ldc,
		const char *file, const char * line );

void phidgemm_ (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const double *alpha,
		const double *A, const int *lda, const double *B,
		const int *ldb, const double *beta, double *C, const int *ldc,
		const char *file, const char * line );

void phidgemm_specialk_ (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const double *alpha,
		const double *A, const int *lda, const double *B,
		const int *ldb, const double *beta, double *C, const int *ldc,
		const char *file, const char * line );

void phicgemm_ (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const cuComplex *alpha,
		const cuComplex *A, const int *lda, const cuComplex *B,
		const int *ldb, const cuComplex *beta, cuComplex *C,
		const int *ldc, const char *file, const char * line );

void phizgemm_ (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const cuDoubleComplex *alpha,
		const cuDoubleComplex *A, const int *lda, const cuDoubleComplex *B,
		const int *ldb, const cuDoubleComplex *beta, cuDoubleComplex *C,
		const int *ldc, const char *file, const char * line );

void phizgemm_specialk_ (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const cuDoubleComplex *alpha,
		const cuDoubleComplex *A, const int *lda, const cuDoubleComplex *B,
		const int *ldb, const cuDoubleComplex *beta, cuDoubleComplex *C,
		const int *ldc, const char *file, const char * line );
#else
void phisgemm_ (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const float *alpha,
		const float *A, const int *lda, const float *B,
		const int *ldb, const float *beta, float *C, const int *ldc);

void phidgemm_ (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const double *alpha,
		const double *A, const int *lda, const double *B,
		const int *ldb, const double *beta, double *C, const int *ldc);

void phidgemm_specialk_ (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const double *alpha,
		const double *A, const int *lda, const double *B,
		const int *ldb, const double *beta, double *C, const int *ldc);

void phicgemm_ (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const cuComplex *alpha,
		const cuComplex *A, const int *lda, const cuComplex *B,
		const int *ldb, const cuComplex *beta, cuComplex *C,
		const int *ldc);

void phizgemm_ (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const cuDoubleComplex *alpha,
		const cuDoubleComplex *A, const int *lda, const cuDoubleComplex *B,
		const int *ldb, const cuDoubleComplex *beta, cuDoubleComplex *C,
		const int *ldc);

void phizgemm_specialk_ (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const cuDoubleComplex *alpha,
		const cuDoubleComplex *A, const int *lda, const cuDoubleComplex *B,
		const int *ldb, const cuDoubleComplex *beta, cuDoubleComplex *C,
		const int *ldc);
#endif

#ifdef __cplusplus
}
#endif

#endif //__PHIGEMM_H__
