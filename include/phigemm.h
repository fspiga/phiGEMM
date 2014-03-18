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

#ifndef __PHIGEMM_H__
#define __PHIGEMM_H__

#include "phigemm_common.h"

#ifdef __cplusplus
extern "C"
{
#endif

/* C interface */

void phiGemmInit( void* dev_ptr, size_t dev_memsize, int deviceToBond, int tag );

void phiGemmShutdown();

int phiGemmIsInit();

void phigemmSetSplitFactor(float split_gemm);

void phiGemmSetAvaiableScratchSpace(int gpu_id, size_t new_dev_memsize);

#if defined(__PHIGEMM_PROFILE)
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

void phiZgemm (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const phiDoubleComplex *alpha,
		const phiDoubleComplex *A, const int *lda, const phiDoubleComplex *B,
		const int *ldb, const phiDoubleComplex *beta, phiDoubleComplex *C,
		const int *ldc, const char *file, const char * line );

void phizgemm_specialK (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const phiDoubleComplex *alpha,
		const phiDoubleComplex *A, const int *lda, const phiDoubleComplex *B,
		const int *ldb, const phiDoubleComplex *beta, phiDoubleComplex *C,
		const int *ldc, const char *file, const char * line );
#else
	void phiDgemm (const char *transa, const char *transb, const int *m,
			const int *n, const int *k, const double *alpha,
			const double *A, const int *lda, const double *B,
			const int *ldb, const double *beta, double *C, const int *ldc);

	void phidgemm_specialK(const char *transa, const char *transb, const int *m,
			const int *n, const int *k, const double *alpha,
			const double *A, const int *lda, const double *B,
			const int *ldb, const double *beta, double *C, const int *ldc);

	void phiZgemm (const char *transa, const char *transb, const int *m,
			const int *n, const int *k, const phiDoubleComplex *alpha,
			const phiDoubleComplex *A, const int *lda, const phiDoubleComplex *B,
			const int *ldb, const phiDoubleComplex *beta, phiDoubleComplex *C,
			const int *ldc);

	void phizgemm_specialK (const char *transa, const char *transb, const int *m,
			const int *n, const int *k, const phiDoubleComplex *alpha,
			const phiDoubleComplex *A, const int *lda, const phiDoubleComplex *B,
			const int *ldb, const phiDoubleComplex *beta, phiDoubleComplex *C,
			const int *ldc);
#endif

/* Fortran interface */

void phigemminit_(void* dev_ptr, size_t dev_memsize, int deviceToBond, int tag );

void phigemmshutdown_();

int phigemmisinit_();

void phigemmsetsplitfactor_(float split_gemm);

#if defined(__PHIGEMM_PROFILE)
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

void phizgemm_ (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const phiDoubleComplex *alpha,
		const phiDoubleComplex *A, const int *lda, const phiDoubleComplex *B,
		const int *ldb, const phiDoubleComplex *beta, phiDoubleComplex *C,
		const int *ldc, const char *file, const char * line );

void phizgemm_specialk_ (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const phiDoubleComplex *alpha,
		const phiDoubleComplex *A, const int *lda, const phiDoubleComplex *B,
		const int *ldb, const phiDoubleComplex *beta, phiDoubleComplex *C,
		const int *ldc, const char *file, const char * line );
#else
void phidgemm_ (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const double *alpha,
		const double *A, const int *lda, const double *B,
		const int *ldb, const double *beta, double *C, const int *ldc);

void phidgemm_specialk_ (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const double *alpha,
		const double *A, const int *lda, const double *B,
		const int *ldb, const double *beta, double *C, const int *ldc);

void phizgemm_ (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const phiDoubleComplex *alpha,
		const phiDoubleComplex *A, const int *lda, const phiDoubleComplex *B,
		const int *ldb, const phiDoubleComplex *beta, phiDoubleComplex *C,
		const int *ldc);

void phizgemm_specialk_ (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const phiDoubleComplex *alpha,
		const phiDoubleComplex *A, const int *lda, const phiDoubleComplex *B,
		const int *ldb, const phiDoubleComplex *beta, phiDoubleComplex *C,
		const int *ldc);
#endif

#ifdef __cplusplus
}
#endif

#endif //__PHIGEMM_H__
