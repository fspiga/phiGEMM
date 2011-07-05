/*
 * Copyright (C) 2010-2011 Irish Centre for High-End Computing (ICHEC)
 *
 * This file is distributed under the terms of the
 * GNU General Public License. See the file `License'
 * in the root directory of the present distribution,
 * or http://www.gnu.org/copyleft/gpl.txt .
 *
 * author(s): Philip Yang   (phi@cs.umd.edu)
 * 			  Filippo Spiga (filippo.spiga@ichec.ie)
 * 			  Ivan Girotto  (ivan.girotto@ichec.ie)
 */

#include "phigemm.h"
#include "phigemm_auxiliary.h"
#include <sys/time.h>

/*
 * SGEMM definitions
 */
#if defined CUDA_TYPE_FLOAT
#define XTYPE float
#ifdef __PHIGEMM_MAGMA
#define cublasGemm magmablas_fermi_sgemm
#else
#define cublasGemm cublasSgemm
#endif
#define gemm_mkl SGEMM_
#define CUBLAS_GEMM phisgemm_
#define CUBLAS_GEMM_MF CUBLAS_SGEMM_MF
#define sgemm CUBLAS_GEMM
#define sgemm_ CUBLAS_GEMM
#define phiSgemm CUBLAS_GEMM

/*
 * DGEMM definitions
 */
#elif defined CUDA_TYPE_DOUBLE
#define XTYPE double
#ifdef __PHIGEMM_MAGMA
#define cublasGemm magmablas_fermi_dgemm
#else
#define cublasGemm cublasDgemm
#endif
#define gemm_mkl DGEMM_
#define CUBLAS_GEMM phidgemm_
#define CUBLAS_GEMM_MF CUBLAS_DGEMM_MF
#define dgemm CUBLAS_GEMM
#define dgemm_ CUBLAS_GEMM
#define phiDgemm CUBLAS_GEMM

/*
 * ZGEMM definitions
 */
#elif defined CUDA_TYPE_COMPLEX
#define XTYPE cuDoubleComplex
#define cublasGemm cublasZgemm
#define gemm_mkl ZGEMM_
#define CUBLAS_GEMM phizgemm_
#define CUBLAS_GEMM_MF CUBLAS_ZGEMM_MF
#define dgemm CUBLAS_GEMM
#define dgemm_ CUBLAS_GEMM
#define phiZgemm CUBLAS_GEMM

#else
#error Missing flag at compile-time
#endif


void CUBLAS_GEMM_MF (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const XTYPE *alpha,
		const XTYPE *A, const int *lda, const XTYPE *B,
		const int *ldb, const XTYPE *beta, XTYPE *C, const int *ldc,
		int is_splitA, float split);

extern phiGemmMemSizes scratch_size;
extern phiGemmMemDevPtr dev_scratch;
extern phiGemmDeviceIds deviceIds;
extern float phiGemmSplitFactor;
extern int phiGemmNumDevices;

#ifdef __PHIGEMM_PROFILE
extern FILE *phiProfileFile;
#endif

#ifdef __PHIGEMM_PROFILE
void CUBLAS_GEMM (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const XTYPE *alpha,
		const XTYPE *A, const int *lda, const XTYPE *B,
		const int *ldb, const XTYPE *beta, XTYPE *C, const int *ldc,
		const char *file, const int line)
#else
void CUBLAS_GEMM (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const XTYPE *alpha,
		const XTYPE *A, const int *lda, const XTYPE *B,
		const int *ldb, const XTYPE *beta, XTYPE *C, const int *ldc)
#endif
{
	double time_call;
	int tmp, p1, p2, m_gpu, n_gpu, k_gpu;
	int a_offset, b_offset, c_offset;
	size_t memsize_gpu, mem_gpu;
	float split;
	static int ground_level = 1;
	int first_call = 0;

	/* determine which matrix is to be splitted */
	int is_splitA = -1;

#ifdef __PHIGEMM_PROFILE
	double start, stop;
#endif

	/* Enabling these checks? */
	//	int err_param = 1;
	//	if ( !transa && ++err_param || !transb && ++err_param || !m && ++err_param || !n && ++err_param ||
	//			!k && ++err_param || !alpha && ++err_param || !A && ++err_param ||
	//			!lda && ++err_param || !B && ++err_param || !ldb && ++err_param ||
	//			!beta && ++err_param || !C && ++err_param || !ldc )
	//	{
	//		fprintf(stderr, "phiGEMM Error: input parameter %d is invalid\n", err_param); fflush(stdout);
	//		exit(EXIT_FAILURE);
	//	}

	/* Enabling these checks? */
	//	if ( (*m) < 1 || (*n) < 1 || (*k) < 1 )
	//	{
	//		fprintf(stderr, "phiGEMM Error: the dimensions m %d, n %d, k %d is not valid\n", *m, *n, *k); fflush(stdout);
	//		exit(EXIT_FAILURE);
	//	}

#ifdef __PHIGEMM_PROFILE
	if ( ground_level) {
		first_call = 1;
		start = phigemm_cclock();
	}
#endif

	/* if the input matrix if pretty small, we will perform the computation on CPU */
	if ( (*n) < 254 && (*m) < 254 && (*k) < 254 )
	{
		gemm_mkl(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta,C, ldc);
		return;
	}

	if ( ground_level && !phiGemmIsInit() )
	{
		fprintf(stderr, "*** phiGEMM *** ERROR *** Missing initialization\n"); fflush(stdout);
		exit(EXIT_FAILURE);
	}

	is_splitA = (*n > *m) ? 0:1;

	/* Assign the split factor */
	split = phiGemmSplitFactor;

	/* smart padding for Fermi & CUDA 3.x - no needed anymore */
	//	m_gpu = ceil(*m/64.0)*64;
	//	n_gpu = ceil(*n/64.0)*64;
	//	k_gpu = ceil(*k/64.0)*64;

	/* smart padding for Tesla & CUDA 3.x  - no needed anymore */
	//	m_gpu = ceil(*m/64.0)*64;
	//	n_gpu = ceil(*n/16.0)*16;
	//	k_gpu = ceil(*k/16.0)*16;

	/* padding seems not required anymore with CUDA 4.x */
	m_gpu = *m;
	n_gpu = *n;
	k_gpu = *k;

	/* recursive splitting */
	/* There is an assumption here: all the cards has the same amount of memory.
	 * This can be not true at all! */
	memsize_gpu = scratch_size[0] * phiGemmNumDevices;

	if ( is_splitA )
	{
		tmp = (*m) * split;
		m_gpu = floor(tmp/64.0)*64;

		mem_gpu = ( m_gpu*k_gpu/phiGemmNumDevices + k_gpu*n_gpu + m_gpu*n_gpu/phiGemmNumDevices ) * sizeof(XTYPE);
		if ( mem_gpu * phiGemmNumDevices > memsize_gpu )
		{
			ground_level = 0;

#ifdef __PHIGEMM_DEBUG
			printf("*** phiGEMM *** matrix size (%lu Bytes) too big to fit in GPU memory (%lu Bytes), splitting A(%d, %d) recursively\n",(unsigned long)mem_gpu, (unsigned long)memsize_gpu, m_gpu, n_gpu);  fflush(stdout);
#endif

			/* this can be improved and be a function of the split factor (NdFilippo) */
			p1 = (*m)/2;
			p2 = (*m) - p1;
#ifdef __PHIGEMM_DEBUG
			fprintf( stdout,"*** phiGEMM *** > SPLIT A: ( %d %d ) %d %d (transa: %c, transb: %c)\n", p1, p2, *n, *k, *transa, *transb); fflush(stdout);
#endif
			a_offset = ( *transa == 'n' || *transa == 'N' )? p1 : ((*lda)*p1);
			c_offset = p1;
#ifdef __PHIGEMM_PROFILE
			CUBLAS_GEMM(transa, transb, &p1, n, k, alpha, A, lda, B, ldb, beta, C, ldc, file, line);
			CUBLAS_GEMM(transa, transb, &p2, n, k, alpha, A + a_offset, lda, B, ldb, beta, C + c_offset, ldc, file, line);
#else
			CUBLAS_GEMM(transa, transb, &p1, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
			CUBLAS_GEMM(transa, transb, &p2, n, k, alpha, A + a_offset, lda, B, ldb, beta, C + c_offset, ldc);
#endif
		} else {
#ifdef __PHIGEMM_DEBUG
			fprintf( stdout,"*** phiGEMM *** > MxM (A): %d, %d, %d (transa: %c, transb: %c)\n", *m, *n, *k, *transa, *transb); fflush(stdout);
#endif
			CUBLAS_GEMM_MF(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, is_splitA, split);
		}

	} else {
		tmp = (*n) * split;
		n_gpu = floor(tmp/64.0)*64;

		size_t mem_gpu = ( m_gpu*k_gpu + k_gpu*n_gpu/phiGemmNumDevices + m_gpu*n_gpu/phiGemmNumDevices ) * sizeof(XTYPE);
		if ( mem_gpu * phiGemmNumDevices > memsize_gpu )
		{
			ground_level = 0;
#ifdef __PHIGEMM_DEBUG
			printf("*** phiGEMM *** matrix size (%lu Bytes) too big to fit in GPU memory (%lu Bytes), splitting B( %d, %d ) recursively\n",
					(unsigned long)mem_gpu, (unsigned long)memsize_gpu, k_gpu, n_gpu); fflush(stdout);
#endif
			p1 = (*n)/2;
			p2 = (*n) - p1;
#ifdef __PHIGEMM_DEBUG
			fprintf( stdout,"*** phiGEMM *** > SPLIT B: %d (%d %d) %d (transa: %c, transb: %c)\n", *m, p1, p2, *k, *transa, *transb); fflush(stdout);
#endif
			b_offset = ( *transb == 'n' || *transb == 'N' )? ((*ldb)*p1) : p1;
			c_offset = (*ldc)*p1;
#ifdef __PHIGEMM_PROFILE
			CUBLAS_GEMM(transa, transb, m, &p1, k, alpha, A, lda, B, ldb, beta, C, ldc, file, line);
			CUBLAS_GEMM(transa, transb, m, &p2, k, alpha, A, lda, B + b_offset, ldb, beta, C + c_offset, ldc, file, line);
#else
			CUBLAS_GEMM(transa, transb, m, &p1, k, alpha, A, lda, B, ldb, beta, C, ldc);
			CUBLAS_GEMM(transa, transb, m, &p2, k, alpha, A, lda, B + b_offset, ldb, beta, C + c_offset, ldc);
#endif
		} else {
#ifdef __PHIGEMM_DEBUG
			fprintf( stdout,"*** phiGEMM *** > MxM (B): %d, %d, %d (transa: %c, transb: %c)\n", *m, *n, *k, *transa, *transb); fflush(stdout);
#endif
			CUBLAS_GEMM_MF(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, is_splitA, split);
		}
	}

#ifdef __PHIGEMM_PROFILE
	if ( first_call) {
		ground_level = 1;
		stop = phigemm_cclock() - start;
		fprintf (phiProfileFile, "[%s:%d]\tm:%d\tn:%d\tk:%d\t[%c%c]\t%g GFlops\n", file, line, *m, *n, *k, *transa, *transb, ( 2.e-9*(*m)*(*n)*(*k)/stop));
	}
#endif

#if !defined __PHIGEMM_PARA
	if ( cudaSetDevice(0) != cudaSuccess) {
		printf("*** phiGEMM *** ERROR *** cudaSetDevice failed!\n");
		exit(EXIT_FAILURE);
	}
#endif
}


void CUBLAS_GEMM_MF (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const XTYPE *alpha,
		const XTYPE *A, const int *lda, const XTYPE *B,
		const int *ldb, const XTYPE *beta, XTYPE *C, const int *ldc,
		int is_splitA, float split)
{
	int iDevice;
	int m_gpu[NSTREAM_PER_DEVICE *MAX_GPUS], n_gpu[NSTREAM_PER_DEVICE *MAX_GPUS], k_gpu[NSTREAM_PER_DEVICE *MAX_GPUS];
	int m_cpu, n_cpu, k_cpu;
	int m_h2d[NSTREAM_PER_DEVICE *MAX_GPUS], n_h2d[NSTREAM_PER_DEVICE *MAX_GPUS], k_h2d[NSTREAM_PER_DEVICE *MAX_GPUS]; // the size of matrix transfered from main memory
	size_t a_offset, b_offset, c_offset;
	size_t a_offset_gpu[NSTREAM_PER_DEVICE *MAX_GPUS], b_offset_gpu[NSTREAM_PER_DEVICE *MAX_GPUS], c_offset_gpu[NSTREAM_PER_DEVICE *MAX_GPUS];

	size_t shiftA, shiftB, shiftC;

	XTYPE *devPtrA[NSTREAM_PER_DEVICE *MAX_GPUS], *devPtrB[NSTREAM_PER_DEVICE *MAX_GPUS], *devPtrC[NSTREAM_PER_DEVICE *MAX_GPUS];
	cublasStatus_t status;
	cudaError_t cudaErr;

#ifdef __PHIGEMM_DEBUG
	int j,i;

	/* timing using CUDA events */
	cudaEvent_t eventPointers[phiGemmNumDevices * NSTREAM_PER_DEVICE][10];

	/* timing using CPU clocks */
	double start_lunchers, start_mkl, start_sync, start_total;
	double stop_lunchers, stop_mkl, stop_sync, stop_total;

	start_total = phigemm_cclock();

#endif

	/* check if the matrices are transposed */
	int is_transa = 0;
	int is_transb = 0;
	if ( (*transa != 'n') && (*transa != 'N') )
		is_transa = 1;
	if ( (*transb != 'n') && (*transb != 'N') )
		is_transb = 1;

	int tmp;
	int step;
	int residual;

	/* split A only */
	if (is_splitA)
	{
		tmp = (*m) * split;
		tmp = floor(tmp/64.0)*64;
		m_cpu = *m - tmp;

		for (iDevice = 0; iDevice < phiGemmNumDevices * NSTREAM_PER_DEVICE; iDevice++) {

			step = (int) (tmp / ( phiGemmNumDevices * NSTREAM_PER_DEVICE ) );
			residual =  tmp - phiGemmNumDevices * NSTREAM_PER_DEVICE *step;

			n_h2d[iDevice] = n_gpu[iDevice] = n_cpu = *n;
			k_h2d[iDevice] = k_gpu[iDevice] = k_cpu = *k;
			m_h2d[iDevice] = m_gpu[iDevice] = (iDevice==0) ? step + residual : step;

			if ( is_transa )
				a_offset_gpu[iDevice] = m_gpu[iDevice] * (*lda);
			else
				a_offset_gpu[iDevice] = m_gpu[iDevice] ;

			b_offset_gpu[iDevice] = 0;
			c_offset_gpu[iDevice] = m_gpu[iDevice] ;
		}

		if ( is_transa )
			a_offset = tmp * (*lda);
		else
			a_offset = tmp;

		b_offset = 0;
		c_offset = tmp;

	} else {

		tmp = (*n) * split ;
		tmp = floor(tmp/64.0)*64;
		n_cpu = *n - tmp;

		for (iDevice = 0; iDevice < phiGemmNumDevices * NSTREAM_PER_DEVICE; iDevice++) {

			step = tmp / phiGemmNumDevices * NSTREAM_PER_DEVICE;
			residual =  tmp - phiGemmNumDevices * NSTREAM_PER_DEVICE * step;

			k_h2d[iDevice] = k_gpu[iDevice] = k_cpu = *k;
			m_h2d[iDevice] = m_gpu[iDevice] = m_cpu = *m;
			n_h2d[iDevice] = n_gpu[iDevice] = (iDevice==0) ? step + residual : step;

			if ( is_transb )
				b_offset_gpu[iDevice] = n_gpu[iDevice];
			else
				b_offset_gpu[iDevice] = (*ldb) * n_gpu[iDevice] ;

			a_offset_gpu[iDevice] = 0;
			c_offset_gpu[iDevice] = (*ldc) * n_gpu[iDevice] ;
		}

		if ( is_transb )
			b_offset = tmp;
		else
			b_offset = (*ldb)* tmp;

		a_offset = 0;
		c_offset = (*ldc) * tmp ;
	}

	/* unused... */
#ifdef __CUDA_MATRIX_PADDING
#ifdef __PHIGEMM_MAGMA
	m_gpu[iDevice] = ceil(m_h2d[iDevice]/64.0)*64;
	n_gpu[iDevice] = ceil(n_h2d[iDevice]/64.0)*64;
	k_gpu[iDevice] = ceil(k_h2d[iDevice]/64.0)*64;
#else
	m_gpu[iDevice] = ceil(m_h2d[iDevice]/64.0)*64;
	n_gpu[iDevice] = ceil(n_h2d[iDevice]/16.0)*16;
	k_gpu[iDevice] = ceil(k_h2d[iDevice]/16.0)*16;
#endif
#endif	

	shiftA = 0;
	shiftB = 0;
	shiftC = 0;

#ifdef __PHIGEMM_DEBUG
	start_lunchers = phigemm_cclock();
#endif

	for (iDevice = 0; iDevice < phiGemmNumDevices * NSTREAM_PER_DEVICE; iDevice++) {

		cudaSetDevice(deviceIds[iDevice % phiGemmNumDevices]);

#ifdef __PHIGEMM_DEBUG
		for (j = 0; j < 10; j++)
			cudaEventCreate(&(eventPointers[iDevice % phiGemmNumDevices][j]));
#endif

		devPtrA[iDevice]=(XTYPE *)(dev_scratch[iDevice]);

		if ( is_transa ) {

#ifdef __PHIGEMM_DEBUG
			cudaEventRecord(eventPointers[iDevice][0], phiStreams[iDevice] );
#endif

#ifdef __PHIGEMM_MEM_ASYNC
			status = cublasSetMatrixAsync (k_h2d[iDevice], m_h2d[iDevice], sizeof(A[0]), A+shiftA, *lda, devPtrA[iDevice], k_gpu[iDevice], phiStreams[iDevice]);
#else
			status = cublasSetMatrix (k_h2d[iDevice], m_h2d[iDevice], sizeof(A[0]), A+shiftA, *lda, devPtrA[iDevice], k_gpu[iDevice]);
#endif

#ifdef __PHIGEMM_DEBUG
			cudaEventRecord(eventPointers[iDevice][1], phiStreams[iDevice] );
#endif

			shiftA += m_h2d[iDevice] * (*lda);
		} else {

#ifdef __PHIGEMM_DEBUG
			cudaEventRecord(eventPointers[iDevice][0], phiStreams[iDevice] );
#endif

#ifdef __PHIGEMM_MEM_ASYNC
			status = cublasSetMatrixAsync (m_h2d[iDevice], k_h2d[iDevice], sizeof(A[0]), A+shiftA, *lda, devPtrA[iDevice], m_gpu[iDevice], phiStreams[iDevice]);
#else
			status = cublasSetMatrix (m_h2d[iDevice], k_h2d[iDevice], sizeof(A[0]), A+shiftA, *lda, devPtrA[iDevice], m_gpu[iDevice]);
#endif

#ifdef __PHIGEMM_DEBUG
			cudaEventRecord(eventPointers[iDevice][1], phiStreams[iDevice] );
#endif

			shiftA += m_h2d[iDevice];
		}

		devPtrB[iDevice] = devPtrA[iDevice] + m_gpu[iDevice] * k_gpu[iDevice];
		if ( is_transb ) {

#ifdef __PHIGEMM_DEBUG
			cudaEventRecord(eventPointers[iDevice][2], phiStreams[iDevice] );
#endif

#ifdef __PHIGEMM_MEM_ASYNC
			status = cublasSetMatrixAsync (n_h2d[iDevice], k_h2d[iDevice], sizeof(B[0]), B+shiftB, *ldb, devPtrB[iDevice], n_gpu[iDevice], phiStreams[iDevice]);
#else
			status = cublasSetMatrix (n_h2d[iDevice], k_h2d[iDevice], sizeof(B[0]), B+shiftB, *ldb, devPtrB[iDevice], n_gpu[iDevice]);
#endif

#ifdef __PHIGEMM_DEBUG
			cudaEventRecord(eventPointers[iDevice][3], phiStreams[iDevice] );
#endif

			shiftB += n_h2d[iDevice];
		} else {

#ifdef __PHIGEMM_DEBUG
			cudaEventRecord(eventPointers[iDevice][2], phiStreams[iDevice] );
#endif

#ifdef __PHIGEMM_MEM_ASYNC
			status = cublasSetMatrixAsync (k_h2d[iDevice], n_h2d[iDevice], sizeof(B[0]), B+shiftB, *ldb, devPtrB[iDevice], k_gpu[iDevice], phiStreams[iDevice]);
#else
			status = cublasSetMatrix (k_h2d[iDevice], n_h2d[iDevice], sizeof(B[0]), B+shiftB, *ldb, devPtrB[iDevice], k_gpu[iDevice]);
#endif

#ifdef __PHIGEMM_DEBUG
			cudaEventRecord(eventPointers[iDevice][3], phiStreams[iDevice] );
#endif

			shiftB += n_h2d[iDevice] * (*ldb);
		}

		/* set the matrix C to device */
		devPtrC[iDevice] = devPtrB[iDevice] + k_gpu[iDevice] * n_gpu[iDevice];

#ifndef CUDA_TYPE_COMPLEX
		if ( (* beta) != (XTYPE)0.0 ) {
#else
			if ( beta->x != 0.0 || beta->y != 0.0 ) {
#endif

#ifdef __PHIGEMM_DEBUG
				cudaEventRecord(eventPointers[iDevice][4], phiStreams[iDevice] );
#endif

#ifdef __PHIGEMM_MEM_ASYNC
				status = cublasSetMatrixAsync (m_h2d[iDevice], n_h2d[iDevice], sizeof(C[0]), C+shiftC, *ldc, devPtrC[iDevice], m_gpu[iDevice], phiStreams[iDevice]);
#else
				status = cublasSetMatrix (m_h2d[iDevice], n_h2d[iDevice], sizeof(C[0]), C+shiftC, *ldc, devPtrC[iDevice], m_gpu[iDevice]);
#endif

#ifdef __PHIGEMM_DEBUG
				cudaEventRecord(eventPointers[iDevice][5], phiStreams[iDevice] );
#endif
			}

			int gpu_lda = m_gpu[iDevice];
			int gpu_ldb = k_gpu[iDevice];

			if ( is_transa ) gpu_lda = k_gpu[iDevice];
			if ( is_transb ) gpu_ldb = n_gpu[iDevice];

			cublasOperation_t cu_transa, cu_transb;
			cu_transa =  ( (*transa == 'c') || (*transa == 'C') ) ? CUBLAS_OP_C : -1;
			cu_transa =  ( (*transa == 't') || (*transa == 'T') ) ? CUBLAS_OP_T : cu_transa;
			cu_transa =  ( (*transa == 'n') || (*transa == 'N') ) ? CUBLAS_OP_N : cu_transa;
			cu_transb =  ( (*transb == 'c') || (*transb == 'C') ) ? CUBLAS_OP_C : -1;
			cu_transb =  ( (*transb == 't') || (*transb == 'T') ) ? CUBLAS_OP_T : cu_transb;
			cu_transb =  ( (*transb == 'n') || (*transb == 'N') ) ? CUBLAS_OP_N : cu_transb;

#ifdef __PHIGEMM_DEBUG
			cudaEventRecord(eventPointers[iDevice][6], phiStreams[iDevice] );
#endif

			cublasGemm (phiHandles[ iDevice ], cu_transa, cu_transb, m_gpu[iDevice], n_gpu[iDevice], k_gpu[iDevice], alpha, devPtrA[iDevice], gpu_lda, devPtrB[iDevice], gpu_ldb, beta, devPtrC[iDevice], m_gpu[iDevice]);

#ifdef __PHIGEMM_DEBUG
			cudaEventRecord(eventPointers[iDevice][7], phiStreams[iDevice] );
#endif

			//			if (is_splitA) {
			//				shiftB = 0;
			//				shiftC += m_h2d[iDevice]; //Da cambiare splitB
			//			} else {
			//				shiftA = 0;
			//				shiftC += n_h2d[iDevice] * (*ldc);
			//			}
			//		}

			//		gemm_mkl(transa, transb, &m_cpu, &n_cpu, &k_cpu, alpha, A+a_offset, lda, B+b_offset, ldb, beta, C+c_offset, ldc);

			//shiftC = 0;
			//			for (iDevice = 0; iDevice < phiGemmNumDevices * NSTREAM_PER_DEVICE; iDevice++) {
			//		    cudaSetDevice(iDevice % phiGemmNumDevices);

#ifdef __PHIGEMM_DEBUG
			cudaEventRecord(eventPointers[iDevice][8], phiStreams[iDevice] );
#endif

#ifdef __PHIGEMM_MEM_ASYNC
			status = cublasGetMatrixAsync (m_h2d[iDevice], n_h2d[iDevice], sizeof(C[0]), devPtrC[iDevice], m_gpu[iDevice], C+shiftC, *ldc, phiStreams[iDevice]);
#else
			status = cublasGetMatrix (m_h2d[iDevice], n_h2d[iDevice], sizeof(C[0]), devPtrC[iDevice], m_gpu[iDevice], C+shiftC, *ldc);
#endif

#ifdef __PHIGEMM_DEBUG
			cudaEventRecord(eventPointers[iDevice][9], phiStreams[iDevice] );
#endif

			//			if (is_splitA)
			//				shiftC += m_h2d[iDevice];
			//			else
			//				shiftC += n_h2d[iDevice] * (*ldc);

			if (is_splitA) {
				shiftB = 0;
				shiftC += m_h2d[iDevice]; //Da cambiare splitB
			} else {
				shiftA = 0;
				shiftC += n_h2d[iDevice] * (*ldc);
			}

			if (status != CUBLAS_STATUS_SUCCESS) {
				fprintf (stderr, "!!!! GPU %d: device access error (D2H C) %d\n", iDevice, status); fflush(stderr);
			}
		}
#ifdef __PHIGEMM_DEBUG
		stop_lunchers = phigemm_cclock();
		start_mkl = phigemm_cclock();
#endif

		gemm_mkl(transa, transb, &m_cpu, &n_cpu, &k_cpu, alpha, A+a_offset, lda, B+b_offset, ldb, beta, C+c_offset, ldc);


#ifdef __PHIGEMM_DEBUG
		stop_mkl= phigemm_cclock();
		start_sync = phigemm_cclock();
#endif

		for (iDevice = 0; iDevice < phiGemmNumDevices * NSTREAM_PER_DEVICE; iDevice++) {

			cudaSetDevice(deviceIds[iDevice % phiGemmNumDevices]);

#ifdef __PHIGEMM_MEM_ASYNC
			cudaErr = cudaStreamSynchronize( phiStreams[ iDevice ] );
#else
			cudaErr = cudaDeviceSynchronize();
#endif
			if (cudaErr != cudaSuccess) {
				printf ( "!!!! 4 - cudaDeviceSynchronize error (C) %d\n", cudaErr); fflush(stdout);
			}
		}

#ifdef __PHIGEMM_DEBUG
		stop_sync= phigemm_cclock();
		stop_total = phigemm_cclock();
#endif

		/* is it necessary? no multi-GPU this part! */
#ifdef __CUDA_MATRIX_PADDING
		cudaMemset(dev_scratch, 0, sizeof(C[iDevice])*(m_gpu[iDevice]*k_gpu[iDevice] + k_gpu[iDevice]*n_gpu[iDevice] + m_gpu[iDevice]*n_gpu[iDevice]));

		m_gpu[iDevice] = m_h2d[iDevice];
		n_gpu[iDevice] = n_h2d[iDevice];
		k_gpu[iDevice] = k_h2d[iDevice];
#endif  

#ifdef __PHIGEMM_DEBUG

		float time_temp, time_mem_h2d, time_dgemm_cuda, time_mem_d2h;
		double time_total = stop_total - start_total;
		double time_lunchers = stop_lunchers - start_lunchers;
		double time_sync = stop_sync - start_sync;
		double time_mkl = stop_mkl - start_mkl;
		double unbalance;

		for (iDevice = 0; iDevice < phiGemmNumDevices * NSTREAM_PER_DEVICE; iDevice++) {

			cudaSetDevice(deviceIds[iDevice % phiGemmNumDevices]);

			/* H2D */
			time_mem_h2d = 0.0;
			cudaEventElapsedTime( &time_temp, eventPointers[iDevice][0], eventPointers[iDevice][1] );
			time_mem_h2d += (time_temp / 1000);
			cudaEventElapsedTime( &time_temp, eventPointers[iDevice][2], eventPointers[iDevice][3] );
			time_mem_h2d += (time_temp / 1000);
#ifndef CUDA_TYPE_COMPLEX
			if ( (* beta) != (XTYPE)0.0 ) {
#else
				if ( beta->x != 0.0 || beta->y != 0.0 ) {
#endif
					cudaEventElapsedTime( &time_temp, eventPointers[iDevice][4], eventPointers[iDevice][5] );
					time_mem_h2d += (time_temp / 1000);
				}

				/* CUBLAS*/
				time_dgemm_cuda = 0.0;
				cudaEventElapsedTime( &time_temp, eventPointers[iDevice][6], eventPointers[iDevice][7] );
				time_dgemm_cuda += (time_temp / 1000);

				/* D2H */
				time_mem_d2h = 0.0;
				cudaEventElapsedTime( &time_temp, eventPointers[iDevice][8], eventPointers[iDevice][9] );
				time_mem_d2h += (time_temp / 1000);

				/* For best split, the time to asynchronously move data to device and compute the MxM should be equal
				 * to the time that CPU spent to perform its portion of the GEMM.
				 * NOTE: if (unbalance > 0) the CPU has too less work to do (and the GPU too much)
				 * 		 if (unbalance < 0) the GPU has too less work to do (and the CPU too much)
				 * */
				unbalance = (time_mem_h2d + time_dgemm_cuda + time_mem_d2h) - time_mkl;

				if ( is_splitA ) {
					printf ("[STATS GPU%d] %d (%d %d, %5.4f) %d %d ~ H2D:%9.6fs (%6.4fGB/s) MKL:%9.6fs (%5.4fGflops) CUBLAS: %9.6fs (%7.4fGflops) D2H:%9.6fs (%6.4fGb/s) ~ BALANCE: %9.6fs ~ Total: %9.6fs (%7.4fGflops)\n",
							iDevice % phiGemmNumDevices,
							*m,
							m_gpu[iDevice],
							m_cpu,
							split,
							*k,
							*n,
							time_mem_h2d,
							(k_gpu[iDevice]*(m_gpu[iDevice]+n_gpu[iDevice])+m_gpu[iDevice]*n_gpu[iDevice])/time_mem_h2d/(128*1024*1024),
							time_mkl,
							2.e-9* m_cpu *(int)(*n)*(int)(*k)/time_mkl,
							time_dgemm_cuda,
							2.e-9* m_gpu[iDevice] *(int)(*n)*(int)(*k)/time_dgemm_cuda,
							time_mem_d2h,
							m_gpu[iDevice]*n_gpu[iDevice]/time_mem_d2h/(128*1024*1024),
							unbalance,
							time_total,
							2.e-9* (int)(*m) *(int)(*n)*(int)(*k)/time_total);

				} else {
					printf ("[STATS GPU%d] %d %d %d (%d %d, %5.4f) ~ H2D:%9.6fs (%6.4fGB/s) MKL:%9.6fs (%5.4fGflops) CUBLAS: %9.6fs (%7.4fGflops) D2H:%9.6fs (%6.4fGb/s) ~ BALANCE: %9.6fs~ Total: %9.6fs (%7.4fGflops)\n",
							iDevice % phiGemmNumDevices,
							*m,
							*k,
							*n,
							n_gpu[iDevice],
							n_cpu,
							split,
							time_mem_h2d,
							(k_gpu[iDevice]*(m_gpu[iDevice]+n_gpu[iDevice])+m_gpu[iDevice]*n_gpu[iDevice])/time_mem_h2d/(128*1024*1024),
							time_mkl,
							2.e-9* (int)(*m) * n_cpu *(int)(*k)/time_mkl,
							time_dgemm_cuda,
							2.e-9* (int)(*m) * n_gpu[iDevice] *(int)(*k)/time_dgemm_cuda,
							time_mem_d2h,
							m_gpu[iDevice]*n_gpu[iDevice]/time_mem_d2h/(128*1024*1024),
							unbalance,
							time_total,
							2.e-9* (int)(*m) *(int)(*n)*(int)(*k)/time_total);
				}

			}

			/* Destroy CUDA events */
			for (i = 0; i < phiGemmNumDevices * NSTREAM_PER_DEVICE; i++) {
				cudaSetDevice(deviceIds[i % phiGemmNumDevices]);
				for (j = 0; j < 10; j++)
					cudaEventDestroy(eventPointers[i][j]);
			}
#endif
		}
