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

#include "phigemm.h"
#include "phigemm_auxiliary.h"

#define PHIGEMM_FLOPS(m, n, k) (  6 * GEMM_MUL(m, n, k) +  2 * GEMM_ADD(m, n, k))

#define gpuGemm cublasZgemm
#define gemm_mkl zgemm_
#define PHIGEMM_M phizgemm_
#define phiZgemm PHIGEMM_M

#if defined(__PHIGEMM_PROFILE)
void PHIGEMM_ZGEMM_MF(const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const phiDoubleComplex *alpha,
		const phiDoubleComplex *A, const int *lda, const phiDoubleComplex *B,
		const int *ldb, const phiDoubleComplex *beta, phiDoubleComplex *C, const int *ldc,
		int is_splitA, float split,
		const char *file, const char * line);
#else
void PHIGEMM_ZGEMM_MF(const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const phiDoubleComplex *alpha,
		const phiDoubleComplex *A, const int *lda, const phiDoubleComplex *B,
		const int *ldb, const phiDoubleComplex *beta, phiDoubleComplex *C, const int *ldc,
		int is_splitA, float split);
#endif

#if defined(__PHIGEMM_PROFILE)
void PHIGEMM_M (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const phiDoubleComplex *alpha,
		const phiDoubleComplex *A, const int *lda, const phiDoubleComplex *B,
		const int *ldb, const phiDoubleComplex *beta, phiDoubleComplex *C, const int *ldc,
		const char *file, const char * line)
#else
void PHIGEMM_M (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const phiDoubleComplex *alpha,
		const phiDoubleComplex *A, const int *lda, const phiDoubleComplex *B,
		const int *ldb, const phiDoubleComplex *beta, phiDoubleComplex *C, const int *ldc)
#endif
{
	double time_call;
	int tmp, p1, p2, select_case;
	int a_offset, b_offset, c_offset;
	size_t memsize_gpu, mem_gpu;
	float split = -1;
	static int ground_level = 1;
	static int splitting_level = 0;
	int first_call = 0;
	int local_init = 0;

	/* determine which matrix has to be split */
	int is_splitA = -1;
	int is_specialK = -1;

#if defined(__PHIGEMM_PROFILE)
	double start, stop;
#endif

	if ( ground_level) {
		first_call = 1;
		splitting_level = 0;
#if defined(__PHIGEMM_PROFILE)
		start = phigemm_cclock();
#endif
	}

	split = myPhiGemmTng.split;

	if (split == 1.0) {
		select_case = 2; // GPU ONLY
	} else if (split == 0.0) {
		select_case = 0; // CPU ONLY
	} else {
		if ( ground_level  ) {
			if (!phiGemmIsInit() ) {
				fprintf(stderr, "*** phiGEMM *** ERROR *** Missing initialization. Do CPU-only.\n"); fflush(stdout);
				select_case = 0;
			} else {
				if ( !phiGemmIsInternalMemAlloc() && !phiGemmIsExternalMemAlloc()  )
				{
					// Memory has not been allocated even if phiGEMM has been initialized.
					// Perform memory allocation before any operation!
					phiGemmInitMemory(NULL);
					//phiGemmInitScratchMemory();
				}
				select_case = cpuGPUheuristic( (*m), (*n), (*k));
			}
		}
	}

	switch (select_case)
	{
	case 0:
		ground_level = 0;

		// cpuGPUheuristic(...) = 0 >> CPU-only
		gemm_mkl(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta,C, ldc);

		break;

	case 1:
		ground_level = 0;

		if ( ground_level && !phiGemmIsExternalMemAlloc() ) {
			/* init the memory */
		}

#if defined(__PHIGEMM_DEBUG_2)
		fprintf( stdout,"[PHIGEMM_DEBUG][2] SPECIAL-K (%d, %d, %d) \n", *m, *n, *k); fflush(stdout);
#endif

		// cpuGPUheuristic(...) = 0 >> SPECIAL-K
#if defined(__PHIGEMM_PROFILE)
		phizgemm_specialK( transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, file, line);
#else
		phizgemm_specialK( transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif

		if ( ground_level && phiGemmIsInternalMemAlloc() ) {
			/* free the memory */
		}
		break;

	case 2:
		// cpuGPUheuristic(...) = 0 >> CPU+GPU
		is_splitA = (*n > *m) ? 0:1;

		/* recursive splitting */
		/* There is an assumption here: all the cards has the same amount of memory.
		 * This can be not true at all! */
		memsize_gpu = myPhiGemmHdl.smem[0] * myPhiGemmEnv.numDevices;

		if ( is_splitA ) {
			mem_gpu = memOccupancy(is_splitA, split, *m, *n, *k) * sizeof(phiDoubleComplex);

			if ( mem_gpu * myPhiGemmEnv.numDevices > memsize_gpu )
			{
				ground_level = 0;

				bestFit(is_splitA, split, *m, *n, *k, sizeof(phiDoubleComplex), &p1, &p2);

				a_offset = ( *transa == 'n' || *transa == 'N' )? p1 : ((*lda)*p1);
				c_offset = p1;

				splitting_level++;

#if defined(__PHIGEMM_PROFILE)
				PHIGEMM_M(transa, transb, &p1, n, k, alpha, A, lda, B, ldb, beta, C, ldc, file, line);
				PHIGEMM_M(transa, transb, &p2, n, k, alpha, A + a_offset, lda, B, ldb, beta, C + c_offset, ldc, file, line);
#else
				PHIGEMM_M(transa, transb, &p1, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
				PHIGEMM_M(transa, transb, &p2, n, k, alpha, A + a_offset, lda, B, ldb, beta, C + c_offset, ldc);
#endif
				splitting_level--;
			} else {
#if defined(__PHIGEMM_PROFILE)
				PHIGEMM_ZGEMM_MF(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, is_splitA, split, file, line);
#else
				PHIGEMM_ZGEMM_MF(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, is_splitA, split);
#endif
			}
		} else {
			mem_gpu = memOccupancy(is_splitA, split, *m, *n, *k) * sizeof(phiDoubleComplex);

			if ( mem_gpu * myPhiGemmEnv.numDevices > memsize_gpu )
			{
				ground_level = 0;
				splitting_level++;

				bestFit(is_splitA, split, *m, *n, *k, sizeof(phiDoubleComplex), &p1, &p2);

				b_offset = ( *transb == 'n' || *transb == 'N' )? ((*ldb)*p1) : p1;
				c_offset = (*ldc)*p1;

#if defined(__PHIGEMM_PROFILE)
				PHIGEMM_M(transa, transb, m, &p1, k, alpha, A, lda, B, ldb, beta, C, ldc, file, line);
				PHIGEMM_M(transa, transb, m, &p2, k, alpha, A, lda, B + b_offset, ldb, beta, C + c_offset, ldc, file, line);
#else
				PHIGEMM_M(transa, transb, m, &p1, k, alpha, A, lda, B, ldb, beta, C, ldc);
				PHIGEMM_M(transa, transb, m, &p2, k, alpha, A, lda, B + b_offset, ldb, beta, C + c_offset, ldc);
#endif

				splitting_level--;
			} else {
#if defined(__PHIGEMM_PROFILE)
				PHIGEMM_ZGEMM_MF(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, is_splitA, split, file, line);
#else
				PHIGEMM_ZGEMM_MF(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, is_splitA, split);
#endif
			}
		}
		break;
	}

	if ( first_call) {
		ground_level = 1;
		first_call = 0;
		splitting_level = 0;

		if ( cudaSetDevice(myPhiGemmHdl.devId[0]) != cudaSuccess) {
			printf("*** phiGEMM *** ERROR *** cudaSetDevice failed!\n");
			exit(EXIT_FAILURE);
		}

#if defined(__PHIGEMM_PROFILE)
		stop = phigemm_cclock() - start;
		/* Comma-Separated Value (csv) format:
		 * file, line, nGPU, nThreads, transA, transB, m, n, k, time, GFlops */
		fprintf (myPhiGemmEnv.profileFile, "%s, %s, %c, %c, %d, %d, %d, %10.6f, %10.4f\n", file, line, *transa, *transb, *m, *n, *k, stop, 1.e-6 * PHIGEMM_FLOPS( (double)(*m), (double)(*n), (double)(*k) )/(stop*1000));
#endif

		if ( phiGemmIsInternalMemAlloc() ){
			/* Since phiGemmIsInternalMemAlloc() is True then phiGEMM
			   is still in a initialized state, it means that GPU-process
			   bindings are valid */
			phiGemmShutdown();

#if defined(__PHIGEMM_PROFILE)
			fclose (myPhiGemmEnv.profileFile);
#endif

		}
	}
	return;
}

#if defined(__PHIGEMM_PROFILE)
void PHIGEMM_ZGEMM_MF(const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const phiDoubleComplex *alpha,
		const phiDoubleComplex *A, const int *lda, const phiDoubleComplex *B,
		const int *ldb, const phiDoubleComplex *beta, phiDoubleComplex *C, const int *ldc,
		int is_splitA, float split,
		const char *file, const char * line)
#else
void PHIGEMM_ZGEMM_MF(const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const phiDoubleComplex *alpha,
		const phiDoubleComplex *A, const int *lda, const phiDoubleComplex *B,
		const int *ldb, const phiDoubleComplex *beta, phiDoubleComplex *C, const int *ldc,
		int is_splitA, float split)
#endif
{
	int iDev, i ,j, tmp, step, residual, gpu_lda, gpu_ldb;
	int m_gpu[NSTREAMS *MAX_GPUS], n_gpu[NSTREAMS *MAX_GPUS], k_gpu[NSTREAMS *MAX_GPUS];
	int m_cpu, n_cpu, k_cpu;
	int m_h2d[NSTREAMS *MAX_GPUS], n_h2d[NSTREAMS *MAX_GPUS], k_h2d[NSTREAMS *MAX_GPUS];

	size_t a_offset, b_offset, c_offset;
	size_t a_offset_gpu[NSTREAMS *MAX_GPUS], b_offset_gpu[NSTREAMS *MAX_GPUS], c_offset_gpu[NSTREAMS *MAX_GPUS];
	size_t shiftA, shiftB, shiftC;

	phiDoubleComplex *devPtrA[NSTREAMS *MAX_GPUS], *devPtrB[NSTREAMS *MAX_GPUS], *devPtrC[NSTREAMS *MAX_GPUS];
	cublasStatus_t status;
	cudaError_t cudaErr;

#if defined(__PHIGEMM_DEBUG_2)
	/* timing using CUDA events */
	cudaEvent_t events[myPhiGemmEnv.numDevices * NSTREAMS][__PHIGEMM_EVENTS];

	/* timing using CPU clocks */
	double start_gemm_cpu, start_gemm_total, stop_gemm_cpu, stop_gemm_total;

	start_gemm_total = phigemm_cclock();
#endif

	/* check if the matrices are transposed */
	cublasOperation_t cu_transa, cu_transb;
	int is_transa = 0;
	int is_transb = 0;

	if ( (*transa != 'n') && (*transa != 'N') )	is_transa = 1;
	if ( (*transb != 'n') && (*transb != 'N') ) is_transb = 1;
	cu_transa = ((*transa == 'c')||(*transa == 'C')) ? CUBLAS_OP_C : CUBLAS_OP_N;
	cu_transa = ((*transa == 't')||(*transa == 'T')) ? CUBLAS_OP_T : cu_transa;
	cu_transa = ((*transa == 'n')||(*transa == 'N')) ? CUBLAS_OP_N : cu_transa;
	cu_transb = ((*transb == 'c')||(*transb == 'C')) ? CUBLAS_OP_C : CUBLAS_OP_N;
	cu_transb = ((*transb == 't')||(*transb == 'T')) ? CUBLAS_OP_T : cu_transb;
	cu_transb = ((*transb == 'n')||(*transb == 'N')) ? CUBLAS_OP_N : cu_transb;

	/* split A only */
	if (is_splitA)
	{
		tmp = (*m) * split;
		//if (*m > 128) tmp = floor(tmp/64.0)*64;
		m_cpu = *m - tmp;

		for (iDev = 0; iDev < myPhiGemmEnv.numDevices * NSTREAMS; iDev++) {

			step = (int) (tmp / ( myPhiGemmEnv.numDevices * NSTREAMS ) );
			residual =  tmp - myPhiGemmEnv.numDevices * NSTREAMS * step;

			n_h2d[iDev] = n_gpu[iDev] = n_cpu = *n;
			k_h2d[iDev] = k_gpu[iDev] = k_cpu = *k;
			m_h2d[iDev] = m_gpu[iDev] = (iDev==0) ? step + residual : step;

			if ( is_transa )
				a_offset_gpu[iDev] = m_gpu[iDev] * (*lda);
			else
				a_offset_gpu[iDev] = m_gpu[iDev] ;

			b_offset_gpu[iDev] = 0;
			c_offset_gpu[iDev] = m_gpu[iDev] ;
		}

		if ( is_transa )
			a_offset = tmp * (*lda);
		else
			a_offset = tmp;

		b_offset = 0;
		c_offset = tmp;

	} else {

		tmp = (*n) * split ;
		//if (*n > 128) tmp = floor(tmp/64.0)*64;
		n_cpu = *n - tmp;

		for (iDev = 0; iDev < myPhiGemmEnv.numDevices * NSTREAMS; iDev++) {

			step = tmp / myPhiGemmEnv.numDevices * NSTREAMS;
			residual =  tmp - myPhiGemmEnv.numDevices * NSTREAMS * step;

			k_h2d[iDev] = k_gpu[iDev] = k_cpu = *k;
			m_h2d[iDev] = m_gpu[iDev] = m_cpu = *m;
			n_h2d[iDev] = n_gpu[iDev] = (iDev==0) ? step + residual : step;

			if ( is_transb )
				b_offset_gpu[iDev] = n_gpu[iDev];
			else
				b_offset_gpu[iDev] = (*ldb) * n_gpu[iDev] ;

			a_offset_gpu[iDev] = 0;
			c_offset_gpu[iDev] = (*ldc) * n_gpu[iDev] ;
		}

		if ( is_transb )
			b_offset = tmp;
		else
			b_offset = (*ldb)* tmp;

		a_offset = 0;
		c_offset = (*ldc) * tmp ;
	}

	shiftA = 0;
	shiftB = 0;
	shiftC = 0;

	for (iDev = 0; iDev < myPhiGemmEnv.numDevices * NSTREAMS; iDev++) {

		cudaSetDevice(myPhiGemmHdl.devId[iDev % myPhiGemmEnv.numDevices]);

		devPtrA[iDev]=(phiDoubleComplex *)(myPhiGemmHdl.pmem[iDev]);

#if defined(__PHIGEMM_DEBUG_2)
		for (j = 0; j < __PHIGEMM_EVENTS; j++)
			cudaEventCreate(&(events[iDev % myPhiGemmEnv.numDevices][j]));

		cudaEventRecord(events[iDev][0], myPhiGemmHdl.stream[iDev] );
#endif

		if ( is_transa ) {
			status = cublasSetMatrixAsync (k_h2d[iDev], m_h2d[iDev],
					sizeof(phiDoubleComplex), A+shiftA, *lda, devPtrA[iDev],
					k_gpu[iDev], myPhiGemmHdl.stream[iDev]);
			shiftA += m_h2d[iDev] * (*lda);
		} else {
			status = cublasSetMatrixAsync (m_h2d[iDev], k_h2d[iDev],
					sizeof(phiDoubleComplex), A+shiftA, *lda, devPtrA[iDev],
					m_gpu[iDev], myPhiGemmHdl.stream[iDev]);
			shiftA += m_h2d[iDev];
		}

#if defined(__PHIGEMM_DEBUG_2)
		cudaEventRecord(events[iDev][1], myPhiGemmHdl.stream[iDev] );
#endif

		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf (stderr, "!!!! GPU %d: device access error (H2D A) %d\n", iDev, status); fflush(stderr);
		}

		devPtrB[iDev] = devPtrA[iDev] + m_gpu[iDev] * k_gpu[iDev];
		if ( is_transb ) {
			status = cublasSetMatrixAsync (n_h2d[iDev], k_h2d[iDev],
					sizeof(phiDoubleComplex), B+shiftB, *ldb, devPtrB[iDev],
					n_gpu[iDev], myPhiGemmHdl.stream[iDev]);
			shiftB += n_h2d[iDev];
		} else {
			status = cublasSetMatrixAsync (k_h2d[iDev], n_h2d[iDev],
					sizeof(phiDoubleComplex), B+shiftB, *ldb, devPtrB[iDev],
					k_gpu[iDev], myPhiGemmHdl.stream[iDev]);
			shiftB += n_h2d[iDev] * (*ldb);
		}

		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf (stderr, "!!!! GPU %d: device access error (H2D B) %d\n", iDev, status); fflush(stderr);
		}
		
#if defined(__PHIGEMM_DEBUG_2)
		cudaEventRecord(events[iDev][2], myPhiGemmHdl.stream[iDev] );
#endif

		devPtrC[iDev] = devPtrB[iDev] + k_gpu[iDev] * n_gpu[iDev];
		if ( beta->x != 0.0 || beta->y != 0.0 ){
			status = cublasSetMatrixAsync (m_h2d[iDev], n_h2d[iDev],
					sizeof(phiDoubleComplex), C+shiftC, *ldc, devPtrC[iDev],
					m_gpu[iDev], myPhiGemmHdl.stream[iDev]);



			if (status != CUBLAS_STATUS_SUCCESS) {
				fprintf (stderr, "!!!! GPU %d: device access error (H2D C) %d\n", iDev, status); fflush(stderr);
			}
		}

#if defined(__PHIGEMM_DEBUG_2)
		cudaEventRecord(events[iDev][3], myPhiGemmHdl.stream[iDev] );
#endif

#if defined(__PHIGEMM_PINNED) || defined(__PHIGEMM_MULTI_GPU)

		gpu_lda = m_gpu[iDev];
		gpu_ldb = k_gpu[iDev];

		if ( is_transa ) gpu_lda = k_gpu[iDev];
		if ( is_transb ) gpu_ldb = n_gpu[iDev];

		gpuGemm (myPhiGemmHdl.handle[ iDev ], cu_transa, cu_transb,
				m_gpu[iDev], n_gpu[iDev], k_gpu[iDev],
				alpha, devPtrA[iDev], gpu_lda, devPtrB[iDev], gpu_ldb,
				beta, devPtrC[iDev], m_gpu[iDev]);

#if defined(__PHIGEMM_DEBUG_2)
		cudaEventRecord(events[iDev][4], myPhiGemmHdl.stream[iDev] );
#endif

		status = cublasGetMatrixAsync (m_h2d[iDev], n_h2d[iDev],
				sizeof(phiDoubleComplex), devPtrC[iDev], m_gpu[iDev], C+shiftC,
				*ldc, myPhiGemmHdl.stream[iDev]);
		
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf (stderr, "!!!! GPU %d: device access error (D2H C) %d\n", iDev, status); fflush(stderr);
		}

#if defined(__PHIGEMM_DEBUG_2)
		cudaEventRecord(events[iDev][5], myPhiGemmHdl.stream[iDev] );
#endif

		if (is_splitA) {
			shiftB = 0;
			shiftC += m_h2d[iDev];
		} else {
			shiftA = 0;
			shiftC += n_h2d[iDev] * (*ldc);
		}
	}

#if defined(__PHIGEMM_DEBUG_2)
	start_gemm_cpu = phigemm_cclock();
#endif

	if (split < 1.0) {
		gemm_mkl(transa, transb, &m_cpu, &n_cpu, &k_cpu, alpha, A+a_offset,
			lda, B+b_offset, ldb, beta, C+c_offset, ldc);
	}
	
#if defined(__PHIGEMM_DEBUG_2)
	stop_gemm_cpu= phigemm_cclock();
#endif

	// Sync stream by stream.... we can do better
	for (iDev = 0; iDev < myPhiGemmEnv.numDevices * NSTREAMS; iDev++) {

		cudaSetDevice(myPhiGemmHdl.devId[iDev % myPhiGemmEnv.numDevices]);
		
		cudaErr = (cudaError_t) cudaStreamSynchronize( myPhiGemmHdl.stream[ iDev ] );

		if (cudaErr != cudaSuccess) {
			printf ( "!!!! 4 - cudaDeviceSynchronize error (C) %d\n", cudaErr); fflush(stdout);
		}
	}

#else

		gpu_lda = m_gpu[iDev];
		gpu_ldb = k_gpu[iDev];

		if ( is_transa ) gpu_lda = k_gpu[iDev];
		if ( is_transb ) gpu_ldb = n_gpu[iDev];

		gpuGemm (myPhiGemmHdl.handle[ iDev ], cu_transa, cu_transb, m_gpu[iDev],
				n_gpu[iDev], k_gpu[iDev], alpha, devPtrA[iDev],
				gpu_lda, devPtrB[iDev], gpu_ldb, beta, devPtrC[iDev],
				m_gpu[iDev]);

#if defined(__PHIGEMM_DEBUG_2)
		cudaEventRecord(events[iDev][4], myPhiGemmHdl.stream[iDev] );
#endif

		if (is_splitA) {
			shiftB = 0;
			shiftC += m_h2d[iDev];
		} else {
			shiftA = 0;
			shiftC += n_h2d[iDev] * (*ldc);
		}
	}

#if defined(__PHIGEMM_DEBUG_2)
	start_gemm_cpu = phigemm_cclock();
#endif

	if (split < 1.0) {
		gemm_mkl(transa, transb, &m_cpu, &n_cpu, &k_cpu, alpha, A+a_offset,
			lda, B+b_offset, ldb, beta, C+c_offset, ldc);
	}
	
#if defined(__PHIGEMM_DEBUG_2)
	stop_gemm_cpu= phigemm_cclock();
#endif

	shiftC = 0;
	for (iDev = 0; iDev < myPhiGemmEnv.numDevices * NSTREAMS; iDev++) {
		cudaSetDevice(myPhiGemmHdl.devId[iDev % myPhiGemmEnv.numDevices]);

#if defined(__PHIGEMM_DEBUG_2)
		cudaEventRecord(events[iDev][5], myPhiGemmHdl.stream[iDev] );
#endif

		status = cublasGetMatrixAsync (m_h2d[iDev], n_h2d[iDev],
				sizeof(phiDoubleComplex), devPtrC[iDev], m_gpu[iDev], C+shiftC,
				*ldc, myPhiGemmHdl.stream[iDev]);

		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf (stderr, "!!!! GPU %d: device access error (D2H C) %d\n", iDev, status); fflush(stderr);
		}

#if defined(__PHIGEMM_DEBUG_2)
		cudaEventRecord(events[iDev][6], myPhiGemmHdl.stream[iDev] );
#endif

		if (is_splitA) {
			shiftB = 0;
			shiftC += m_h2d[iDev];
		} else {
			shiftA = 0;
			shiftC += n_h2d[iDev] * (*ldc);
		}

		// Sync stream by stream.... we can do better
		cudaErr = (cudaError_t) cudaStreamSynchronize( myPhiGemmHdl.stream[ iDev ] );
		if (cudaErr != cudaSuccess) {
			printf ( "!!!! 4 - cudaDeviceSynchronize error (C) %d\n", cudaErr); fflush(stdout);
		}
	}
#endif

#if defined(__PHIGEMM_DEBUG_2)
	stop_gemm_total = phigemm_cclock();

	float time_temp, time_mem_h2d, time_gemm_cuda, time_mem_d2h;

	double time_total = stop_gemm_total - start_gemm_total;

	double time_mkl;
	if (split < 1.0)
		time_mkl = stop_gemm_cpu - start_gemm_cpu;
	else
		time_mkl = 0;

	double unbalance;

	for (iDev = 0; iDev < myPhiGemmEnv.numDevices * NSTREAMS; iDev++) {
		cudaSetDevice(myPhiGemmHdl.devId[iDev % myPhiGemmEnv.numDevices]);

		/* H2D */
		time_mem_h2d = 0.0;
		cudaEventElapsedTime( &time_temp, events[iDev][0], events[iDev][1] );
		time_mem_h2d += (time_temp / 1000);
		cudaEventElapsedTime( &time_temp, events[iDev][1], events[iDev][2] );
		time_mem_h2d += (time_temp / 1000);
		
		if ( beta->x != 0.0 || beta->y != 0.0 ){
			cudaEventElapsedTime( &time_temp, events[iDev][2], events[iDev][3] );
			time_mem_h2d += (time_temp / 1000);
		}

		/* CUBLAS*/
		time_gemm_cuda = 0.0;
		cudaEventElapsedTime( &time_temp, events[iDev][3], events[iDev][4] );
		time_gemm_cuda += (time_temp / 1000);

		/* D2H */
		time_mem_d2h = 0.0;
#if defined(__PHIGEMM_PINNED) || defined(__PHIGEMM_MULTI_GPU)
		cudaEventElapsedTime( &time_temp, events[iDev][4], events[iDev][5] );
#else
		cudaEventElapsedTime( &time_temp, events[iDev][5], events[iDev][6] );
#endif
		time_mem_d2h += (time_temp / 1000);

#if defined(__PHIGEMM_PINNED) && defined(__PHIGEMM_MULTI_GPU)
		unbalance = (time_mem_h2d + time_gemm_cuda + time_mem_d2h) - time_mkl;
#elif defined(__PHIGEMM_PINNED)
		unbalance = (time_mem_h2d + time_gemm_cuda) - time_mkl;
#else
		unbalance = time_gemm_cuda - time_mkl;
#endif

		double cpu_flop_computed = 0;

		if ( is_splitA ) {

			if (split < 1.0)
				cpu_flop_computed = 1.e-6 * PHIGEMM_FLOPS( (double)m_cpu, (double)(*n), (double)(*k) )/(time_mkl*1000);

#if defined(__PHIGEMM_PROFILE)
			printf ("[PHIGEMM_DEBUG][2] %s:%s GPU %d - %d (%d %d, %5.4f) %d %d ~ H2D:%9.6fs (%6.4fGB/s) MKL:%9.6fs (%5.4fGflops) CUBLAS: %9.6fs (%7.4fGflops) D2H:%9.6fs (%6.4fGb/s) ~ BALANCE: %9.6fs ~ Total: %9.6fs (%7.4fGflops)\n",
				file, line, iDev % myPhiGemmEnv.numDevices,
#else
			printf ("[PHIGEMM_DEBUG][2] GPU %d - %d (%d %d, %5.4f) %d %d ~ H2D:%9.6fs (%6.4fGB/s) MKL:%9.6fs (%5.4fGflops) CUBLAS: %9.6fs (%7.4fGflops) D2H:%9.6fs (%6.4fGb/s) ~ BALANCE: %9.6fs ~ Total: %9.6fs (%7.4fGflops)\n",
				iDev % myPhiGemmEnv.numDevices,
#endif
			*m,
			m_gpu[iDev],
			m_cpu,
			split,
			*n,
			*k,
			time_mem_h2d,
			(k_gpu[iDev]*(m_gpu[iDev]+n_gpu[iDev])+m_gpu[iDev]*n_gpu[iDev])/time_mem_h2d/(1024*1024*1024/sizeof(phiDoubleComplex)),
			time_mkl,
			cpu_flop_computed,
			time_gemm_cuda,
			1.e-6 * PHIGEMM_FLOPS( (double)m_gpu[iDev], (double)(*n), (double)(*k) )/(time_gemm_cuda*1000),
			time_mem_d2h,
			m_gpu[iDev]*n_gpu[iDev]/time_mem_d2h/(1024*1024*1024/sizeof(phiDoubleComplex)),
			unbalance,
			time_total,
			1.e-6 * PHIGEMM_FLOPS( (double)(*m), (double)(*n), (double)(*k) )/(time_total*1000));
	} else {

		if (split < 1.0)
			cpu_flop_computed = 1.e-6 * PHIGEMM_FLOPS( (double)(*m), (double)n_cpu, (double)(*k) )/(time_mkl*1000);

#if defined(__PHIGEMM_PROFILE)
			printf ("[PHIGEMM_DEBUG][2] %s:%s, GPU %d - %d %d (%d %d, %5.4f) %d ~ H2D:%9.6fs (%6.4fGB/s) MKL:%9.6fs (%5.4fGflops) CUBLAS: %9.6fs (%7.4fGflops) D2H:%9.6fs (%6.4fGb/s) ~ BALANCE: %9.6fs~ Total: %9.6fs (%7.4fGflops)\n",
				file, line, iDev % myPhiGemmEnv.numDevices,
#else
			printf ("[PHIGEMM_DEBUG][2] GPU %d - %d %d (%d %d, %5.4f) %d ~ H2D:%9.6fs (%6.4fGB/s) MKL:%9.6fs (%5.4fGflops) CUBLAS: %9.6fs (%7.4fGflops) D2H:%9.6fs (%6.4fGb/s) ~ BALANCE: %9.6fs~ Total: %9.6fs (%7.4fGflops)\n",
				iDev % myPhiGemmEnv.numDevices,
#endif
			*m,
			*n,
			n_gpu[iDev],
			n_cpu,
			split,
			*k,
			time_mem_h2d,
			(k_gpu[iDev]*(m_gpu[iDev]+n_gpu[iDev])+m_gpu[iDev]*n_gpu[iDev])/time_mem_h2d/(1024*1024*1024/sizeof(phiDoubleComplex)),
			time_mkl,
			cpu_flop_computed,
			time_gemm_cuda,
			1.e-6 * PHIGEMM_FLOPS( (double)(*m), (double)n_gpu[iDev], (double)(*k) )/(time_gemm_cuda*1000),
			time_mem_d2h,
			m_gpu[iDev]*n_gpu[iDev]/time_mem_d2h/(1024*1024*1024/sizeof(phiDoubleComplex)),
			unbalance,
			time_total,
			1.e-6 * PHIGEMM_FLOPS( (double)(*m), (double)(*n), (double)(*k) )/(time_total*1000));
		}
		fflush(stdout);
	}

	/* Destroy CUDA events */
	for (i = 0; i < myPhiGemmEnv.numDevices * NSTREAMS; i++) {
		cudaSetDevice(myPhiGemmHdl.devId[i % myPhiGemmEnv.numDevices]);
		for (j = 0; j < __PHIGEMM_EVENTS; j++)
			cudaEventDestroy(events[i][j]);
	}
#endif
}
