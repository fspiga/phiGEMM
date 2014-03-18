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

#include "phigemm.h"
#include "phigemm_auxiliary.h"

#define PHIGEMM_FLOPS(m, n, k) (      GEMM_MUL(m, n, k) +      GEMM_ADD(m, n, k))

#define gpuGemm cublasDgemm
#define gemm_mkl dgemm_
#define PHIGEMM_M phidgemm_
#define phiDgemm PHIGEMM_M

#if defined(__PHIGEMM_PROFILE)
void PHIGEMM_DGEMM_MF(const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const double *alpha,
		const double *A, const int *lda, const double *B,
		const int *ldb, const double *beta, double *C, const int *ldc,
		int is_splitA, float split,
		const char *file, const char * line);
#else
void PHIGEMM_DGEMM_MF(const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const double *alpha,
		const double *A, const int *lda, const double *B,
		const int *ldb, const double *beta, double *C, const int *ldc,
		int is_splitA, float split);
#endif

#if defined(__PHIGEMM_PROFILE)
void PHIGEMM_M (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const double *alpha,
		const double *A, const int *lda, const double *B,
		const int *ldb, const double *beta, double *C, const int *ldc,
		const char *file, const char * line)
#else
void PHIGEMM_M (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const double *alpha,
		const double *A, const int *lda, const double *B,
		const int *ldb, const double *beta, double *C, const int *ldc)
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

	// printf("\n\n*** phiGEMM *** phiGemmIsInternalMemAlloc() = %d, phiGemmIsExternalMemAlloc() = %d [BEGIN] ***\n",phiGemmIsInternalMemAlloc(), phiGemmIsExternalMemAlloc());

	if ( ground_level) {
		first_call = 1;
		splitting_level = 0;
#if defined(__PHIGEMM_PROFILE)
		start = phigemm_cclock();
#endif
	}

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
			select_case = cpuGPUheuristic( (*m), (*n), (*k), 'd');
		}
	}

	switch (select_case)
	{
	case 0:
		ground_level = 0;

		// cpuGPUheuristic(...) = 0 >> CPU-only
		gemm_mkl(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta,C, ldc);

		break;

#if defined(__PHIGEMM_ENABLE_SPECIALK)
	case 1:
		ground_level = 0;

		// cpuGPUheuristic(...) = 0 >> SPECIAL-K
#if defined(__PHIGEMM_PROFILE)
		phidgemm_specialK( transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, file, line);
#else
		phidgemm_specialK( transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif

		break;
#endif

	case 2:

		is_splitA = (*n > *m) ? 0 : 1;
		split = myPhiGemmHdl.SPLIT;

		/* recursive splitting */
		/* There is an assumption here: all the cards has the same amount of memory.
		 * This can be not true at all! */
		memsize_gpu = myPhiGemmHdl.smem;

		if ( is_splitA ) {

			mem_gpu = memOccupancy(is_splitA, split, *m, *n, *k) * sizeof(double);

			if ( mem_gpu > memsize_gpu )
			{
				ground_level = 0;

				bestFit(is_splitA, split, *m, *n, *k, sizeof(double), &p1, &p2);

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
				PHIGEMM_DGEMM_MF(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, is_splitA, split, file, line);
#else
				PHIGEMM_DGEMM_MF(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, is_splitA, split);
#endif
			}
		} else {

			mem_gpu = memOccupancy(is_splitA, split, *m, *n, *k) * sizeof(double);

			if ( mem_gpu > memsize_gpu )
			{
				ground_level = 0;
				splitting_level++;

				bestFit(is_splitA, split, *m, *n, *k, sizeof(double), &p1, &p2);

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
				PHIGEMM_DGEMM_MF(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, is_splitA, split, file, line);
#else
				PHIGEMM_DGEMM_MF(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, is_splitA, split);
#endif
			}
		}
		break;
	}

	if ( first_call) {
		ground_level = 1;
		first_call = 0;
		splitting_level = 0;

		// ??
		if ( cudaSetDevice(myPhiGemmHdl.devId) != cudaSuccess) {
			printf("*** phiGEMM *** ERROR *** cudaSetDevice failed!\n");
			exit(EXIT_FAILURE);
		}

#if defined(__PHIGEMM_PROFILE)
		stop = phigemm_cclock() - start;
		/* Comma-Separated Value (csv) format:
		 * file, line, transA, transB, m, n, k, (SPECIAL_K) ? -1 : split, time, GFlops */
		fprintf (myPhiGemmEnv.profileFile, "%s, %s, %c, %c, %d, %d, %d, (select_case == 1) ? -1 : split, %10.6f, %10.4f\n", file, line, *transa, *transb, *m, *n, *k, myPhiGemmTng.split, 1.e-6 * PHIGEMM_FLOPS( (double)(*m), (double)(*n), (double)(*k) )/(stop*1000));
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
void PHIGEMM_DGEMM_MF (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const double *alpha,
		const double *A, const int *lda, const double *B,
		const int *ldb, const double *beta, double *C, const int *ldc,
		int is_splitA, float split,
		const char *file, const char * line)
#else
void PHIGEMM_DGEMM_MF (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const double *alpha,
		const double *A, const int *lda, const double *B,
		const int *ldb, const double *beta, double *C, const int *ldc,
		int is_splitA, float split)
#endif
{
	int iDev, i ,j, tmp, step, residual, gpu_lda, gpu_ldb;
	int m_gpu, n_gpu, k_gpu;
	int m_cpu, n_cpu, k_cpu;
	int m_h2d, n_h2d, k_h2d;

	size_t a_offset, b_offset, c_offset;
	size_t a_offset_gpu, b_offset_gpu, c_offset_gpu;
	size_t shiftA, shiftB, shiftC;

	double *devPtrA, *devPtrB, *devPtrC;
	cublasStatus_t status;
	cudaError_t cudaErr;

#if defined(__PHIGEMM_DEBUG)
	/* timing using CUDA events */
	cudaEvent_t events[__PHIGEMM_EVENTS];

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

	// if split == 1 --> all GPU (is it working?)

	/* split A only */
	if (is_splitA)
	{
		tmp = (*m) * split;
		// if (*m > 128) tmp = floor(tmp/64.0)*64;
		m_cpu = *m - tmp;

		step = (int) tmp;
		residual =  tmp - step;

		n_h2d = n_gpu = n_cpu = *n;
		k_h2d = k_gpu = k_cpu = *k;
		m_h2d = m_gpu = step + residual;

		if ( is_transa )
			a_offset_gpu = m_gpu * (*lda);
		else
			a_offset_gpu = m_gpu ;

		b_offset_gpu = 0;
		c_offset_gpu = m_gpu ;

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

		step = tmp ;
		residual =  tmp - step;

		k_h2d = k_gpu = k_cpu = *k;
		m_h2d = m_gpu = m_cpu = *m;
		n_h2d = n_gpu = step + residual;

		if ( is_transb )
			b_offset_gpu = n_gpu;
		else
			b_offset_gpu = (*ldb) * n_gpu ;

		a_offset_gpu = 0;
		c_offset_gpu = (*ldc) * n_gpu ;

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

	cudaSetDevice(myPhiGemmHdl.devId);

	devPtrA=(double *)(myPhiGemmHdl.pmem);

#if defined(__PHIGEMM_DEBUG)
	for (j = 0; j < __PHIGEMM_EVENTS; j++)
		cudaEventCreate(&(events[j]));

	cudaEventRecord(events[0], myPhiGemmHdl.stream );
#endif

	if ( is_transa ) {
		status = cublasSetMatrixAsync (k_h2d, m_h2d,
				sizeof(double), A+shiftA, *lda, devPtrA,
				k_gpu, myPhiGemmHdl.stream);
		shiftA += m_h2d * (*lda);
	} else {
		status = cublasSetMatrixAsync (m_h2d, k_h2d,
				sizeof(double), A+shiftA, *lda, devPtrA,
				m_gpu, myPhiGemmHdl.stream);
		shiftA += m_h2d;
	}

#if defined(__PHIGEMM_DEBUG)
	cudaEventRecord(events[1], myPhiGemmHdl.stream );
#endif

	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! GPU %d: device access error (H2D A) %d\n", iDev, status); fflush(stderr);
	}

	devPtrB = devPtrA + m_gpu * k_gpu;
	if ( is_transb ) {
		status = cublasSetMatrixAsync (n_h2d, k_h2d,
				sizeof(double), B+shiftB, *ldb, devPtrB,
				n_gpu, myPhiGemmHdl.stream);
		shiftB += n_h2d;
	} else {
		status = cublasSetMatrixAsync (k_h2d, n_h2d,
				sizeof(double), B+shiftB, *ldb, devPtrB,
				k_gpu, myPhiGemmHdl.stream);
		shiftB += n_h2d * (*ldb);
	}

#if defined(__PHIGEMM_DEBUG)
	cudaEventRecord(events[2], myPhiGemmHdl.stream );
#endif

	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! GPU %d: device access error (H2D B) %d\n", iDev, status); fflush(stderr);
	}

	/* set the matrix C to device */
	devPtrC = devPtrB + k_gpu * n_gpu;

	if ( (* beta) != (double)0.0 ){
		status = cublasSetMatrixAsync (m_h2d, n_h2d,
				sizeof(double), C+shiftC, *ldc, devPtrC,
				m_gpu, myPhiGemmHdl.stream);

		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf (stderr, "!!!! GPU %d: device access error (H2D C) %d\n", iDev, status); fflush(stderr);
		}
	}

#if defined(__PHIGEMM_DEBUG)
	cudaEventRecord(events[3], myPhiGemmHdl.stream );
#endif

#if defined(__PHIGEMM_PINNED)

	gpu_lda = m_gpu;
	gpu_ldb = k_gpu;

	if ( is_transa ) gpu_lda = k_gpu;
	if ( is_transb ) gpu_ldb = n_gpu;


	gpuGemm (myPhiGemmHdl.handle, cu_transa, cu_transb,
			m_gpu, n_gpu, k_gpu,
			alpha, devPtrA, gpu_lda, devPtrB, gpu_ldb,
			beta, devPtrC, m_gpu);

#if defined(__PHIGEMM_DEBUG)
	cudaEventRecord(events[4], myPhiGemmHdl.stream );
#endif

	status = cublasGetMatrixAsync (m_h2d, n_h2d,
			sizeof(double), devPtrC, m_gpu, C+shiftC,
			*ldc, myPhiGemmHdl.stream);

	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! GPU %d: device access error (D2H C) %d\n", iDev, status); fflush(stderr);
	}

#if defined(__PHIGEMM_DEBUG)
	cudaEventRecord(events[5], myPhiGemmHdl.stream );
#endif

	if (is_splitA) {
		shiftB = 0;
		shiftC += m_h2d;
	} else {
		shiftA = 0;
		shiftC += n_h2d * (*ldc);
	}

#if defined(__PHIGEMM_DEBUG)
	start_gemm_cpu = phigemm_cclock();
#endif

	if (split < 1) {
		gemm_mkl(transa, transb, &m_cpu, &n_cpu, &k_cpu, alpha, A+a_offset,
			lda, B+b_offset, ldb, beta, C+c_offset, ldc);

	}
#if defined(__PHIGEMM_DEBUG)
	stop_gemm_cpu= phigemm_cclock();
#endif

	cudaErr = (cudaError_t) cudaStreamSynchronize( myPhiGemmHdl.stream );

	if (cudaErr != cudaSuccess) {
		printf ( "!!!! 4 - cudaDeviceSynchronize error (C) %d\n", cudaErr); fflush(stdout);
	}

#else

	gpu_lda = m_gpu;
	gpu_ldb = k_gpu;

	if ( is_transa ) gpu_lda = k_gpu;
	if ( is_transb ) gpu_ldb = n_gpu;

	gpuGemm (myPhiGemmHdl.handle, cu_transa, cu_transb, m_gpu,
			n_gpu, k_gpu, alpha, devPtrA,
			gpu_lda, devPtrB, gpu_ldb, beta, devPtrC,
			m_gpu);

#if defined(__PHIGEMM_DEBUG)
	cudaEventRecord(events[4], myPhiGemmHdl.stream );

	start_gemm_cpu = phigemm_cclock();
#endif

	if (split < 1) {
		gemm_mkl(transa, transb, &m_cpu, &n_cpu, &k_cpu, alpha, A+a_offset,
			lda, B+b_offset, ldb, beta, C+c_offset, ldc);
	}

#if defined(__PHIGEMM_DEBUG)
	stop_gemm_cpu= phigemm_cclock();
#endif

	shiftC = 0;
#if defined(__PHIGEMM_DEBUG)
	cudaEventRecord(events[5], myPhiGemmHdl.stream );
#endif

	status = cublasGetMatrixAsync (m_h2d, n_h2d,
			sizeof(double), devPtrC, m_gpu, C+shiftC,
			*ldc, myPhiGemmHdl.stream);

	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! GPU %d: device access error (D2H C) %d\n", iDev, status); fflush(stderr);
	}

#if defined(__PHIGEMM_DEBUG)
	cudaEventRecord(events[6], myPhiGemmHdl.stream );
#endif

	// Sync stream by stream.... we can do better
	cudaErr = (cudaError_t) cudaStreamSynchronize( myPhiGemmHdl.stream );
	if (cudaErr != cudaSuccess) {
		printf ( "!!!! 4 - cudaDeviceSynchronize error (C) %d\n", cudaErr); fflush(stdout);
	}
#endif

#if defined(__PHIGEMM_DEBUG)
	stop_gemm_total = phigemm_cclock();

	float time_temp, time_mem_h2d, time_dgemm_cuda, time_mem_d2h;

	double time_total = stop_gemm_total - start_gemm_total;

	#if !defined(__PHIGEMM_GPUONLY)
	double time_mkl = stop_gemm_cpu - start_gemm_cpu;
	#else
	double time_mkl = 0;
	#endif

	double unbalance;
	float new_split;

	for (iDev = 0; iDev < myPhiGemmEnv.numDevices * NSTREAMS; iDev++) {
		cudaSetDevice(myPhiGemmHdl.devId[iDev % myPhiGemmEnv.numDevices]);

		/* H2D */
		time_mem_h2d = 0.0;
		cudaEventElapsedTime( &time_temp, events[0], events[1] );
		time_mem_h2d += (time_temp / 1000);
		cudaEventElapsedTime( &time_temp, events[1], events[2] );
		time_mem_h2d += (time_temp / 1000);
		if ( (* beta) != (double)0.0 ) {
			cudaEventElapsedTime( &time_temp, events[2], events[3] );
			time_mem_h2d += (time_temp / 1000);
		}

		/* CUBLAS*/
		time_dgemm_cuda = 0.0;
		cudaEventElapsedTime( &time_temp, events[3], events[4] );
		time_dgemm_cuda += (time_temp / 1000);

		/* D2H */
		time_mem_d2h = 0.0;
#if defined(__PHIGEMM_PINNED) || defined(__PHIGEMM_MULTI_GPU)
		cudaEventElapsedTime( &time_temp, events[4], events[5] );
#else
		cudaEventElapsedTime( &time_temp, events[5], events[6] );
#endif
		time_mem_d2h += (time_temp / 1000);

		/* For best split, the time to asynchronously move data to device and compute the MxM should be equal
		 * to the time that CPU spent to perform its portion of the GEMM.
		 * NOTE: if (unbalance > 0) the CPU has too less work to do (and the GPU too much) -> decrease the split
		 * 		 if (unbalance < 0) the GPU has too less work to do (and the CPU too much) -> increase the split
		 * */
#if defined(__PHIGEMM_PINNED) && defined(__PHIGEMM_MULTI_GPU)
		unbalance = (time_mem_h2d + time_dgemm_cuda + time_mem_d2h) - time_mkl;
#elif defined(__PHIGEMM_PINNED)
		unbalance = (time_mem_h2d + time_dgemm_cuda) - time_mkl;
#else
		unbalance = time_dgemm_cuda - time_mkl;
#endif

#if defined(__PHIGEMM_DEBUG)

		if ( is_splitA ) {

#if defined(__PHIGEMM_PROFILE)
			printf ("[PHIGEMM_DEBUG - %s:%s - GPU %d] %d (%d %d, %5.4f) %d %d ~ H2D:%9.6fs (%6.4fGB/s) MKL:%9.6fs (%5.4fGflops) CUBLAS: %9.6fs (%7.4fGflops) D2H:%9.6fs (%6.4fGb/s) ~ BALANCE: %9.6fs ~ Total: %9.6fs (%7.4fGflops)\n",
			file, line, iDev % myPhiGemmEnv.numDevices,
#else
			printf ("[PHIGEMM_DEBUG GPU %d] %d (%d %d, %5.4f) %d %d ~ H2D:%9.6fs (%6.4fGB/s) MKL:%9.6fs (%5.4fGflops) CUBLAS: %9.6fs (%7.4fGflops) D2H:%9.6fs (%6.4fGb/s) ~ BALANCE: %9.6fs ~ Total: %9.6fs (%7.4fGflops)\n",
			iDev % myPhiGemmEnv.numDevices,
#endif
			*m,
			m_gpu,
			m_cpu,
			split,
			*n,
			*k,
			time_mem_h2d,
			(k_gpu*(m_gpu+n_gpu)+m_gpu*n_gpu)/time_mem_h2d/(1024*1024*1024/sizeof(double)),
			time_mkl,
#if !defined(__PHIGEMM_GPUONLY)
			1.e-6 * PHIGEMM_FLOPS( (double)m_cpu, (double)(*n), (double)(*k) )/(time_mkl*1000),
#else
			0.0,
#endif
			time_dgemm_cuda,
			1.e-6 * PHIGEMM_FLOPS( (double)m_gpu, (double)(*n), (double)(*k) )/(time_dgemm_cuda*1000),
			time_mem_d2h,
			m_gpu*n_gpu/time_mem_d2h/(1024*1024*1024/sizeof(double)),
			unbalance,
			time_total,
			1.e-6 * PHIGEMM_FLOPS( (double)(*m), (double)(*n), (double)(*k) )/(time_total*1000));
		} else {
#if defined(__PHIGEMM_PROFILE)
			printf ("[PHIGEMM_DEBUG - %s:%s - GPU %d] %d %d (%d %d, %5.4f) %d ~ H2D:%9.6fs (%6.4fGB/s) MKL:%9.6fs (%5.4fGflops) CUBLAS: %9.6fs (%7.4fGflops) D2H:%9.6fs (%6.4fGb/s) ~ BALANCE: %9.6fs~ Total: %9.6fs (%7.4fGflops)\n",
			file, line, iDev % myPhiGemmEnv.numDevices,
#else
			printf ("[PHIGEMM_DEBUG GPU %d] %d %d (%d %d, %5.4f) %d ~ H2D:%9.6fs (%6.4fGB/s) MKL:%9.6fs (%5.4fGflops) CUBLAS: %9.6fs (%7.4fGflops) D2H:%9.6fs (%6.4fGb/s) ~ BALANCE: %9.6fs~ Total: %9.6fs (%7.4fGflops)\n",
			iDev % myPhiGemmEnv.numDevices,
#endif
			*m,
			*n,
			n_gpu,
			n_cpu,
			split,
			*k,
			time_mem_h2d,
			(k_gpu*(m_gpu+n_gpu)+m_gpu*n_gpu)/time_mem_h2d/(1024*1024*1024/sizeof(double)),
			time_mkl,
#if !defined(__PHIGEMM_GPUONLY)
			1.e-6 * PHIGEMM_FLOPS( (double)(*m), (double)n_cpu, (double)(*k) )/(time_mkl*1000),
#else
			0.0,
#endif
			time_dgemm_cuda,
			1.e-6 * PHIGEMM_FLOPS( (double)(*m), (double)n_gpu, (double)(*k) )/(time_dgemm_cuda*1000),
			time_mem_d2h,
			m_gpu*n_gpu/time_mem_d2h/(1024*1024*1024/sizeof(double)),
			unbalance,
			time_total,
			1.e-6 * PHIGEMM_FLOPS( (double)(*m), (double)(*n), (double)(*k) )/(time_total*1000));
		}
		fflush(stdout);
#endif
	}

	/* Destroy CUDA events */
	for (i = 0; i < myPhiGemmEnv.numDevices * NSTREAMS; i++) {
		cudaSetDevice(myPhiGemmHdl.devId[i % myPhiGemmEnv.numDevices]);
		for (j = 0; j < __PHIGEMM_EVENTS; j++)
			cudaEventDestroy(events[i][j]);
	}
#endif
}
