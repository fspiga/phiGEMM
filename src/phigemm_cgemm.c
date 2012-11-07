/*
 * Copyright (C) 2011-2012 Quantum ESPRESSO Foundation
 * Copyright (C) 2010-2011 Irish Centre for High-End Computing (ICHEC)
 *
 * This file is distributed under the terms of the
 * GNU General Public License. See the file `License'
 * in the root directory of the present distribution,
 * or http://www.gnu.org/copyleft/gpl.txt .
 *
 */

#include "phigemm.h"
#include "phigemm_auxiliary.h"

#define PRECISION_C
#if defined(PRECISION_D) || defined(PRECISION_S)
#define PHIGEMM_FLOPS(m, n, k) (      GEMM_MUL(m, n, k) +      GEMM_ADD(m, n, k))
#else
#define PHIGEMM_FLOPS(m, n, k) (  6 * GEMM_MUL(m, n, k) +  2 * GEMM_ADD(m, n, k))
#endif

#define EVENIZE(x) ( ((x)%2==0) ? (x) : (x)+1 )

#define cublasGemm cublasCgemm
#define gemm_mkl cgemm_
#define PHIGEMM_M phicgemm_
#define phiCgemm PHIGEMM_M


#if defined(__PHIGEMM_PROFILE)
void PHIGEMM_CGEMM_MF(const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const cuComplex *alpha,
		const cuComplex *A, const int *lda, const cuComplex *B,
		const int *ldb, const cuComplex *beta, cuComplex *C, const int *ldc,
		int is_splitA, float split,
		const char *file, const char * line);
#else
void PHIGEMM_CGEMM_MF(const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const cuComplex *alpha,
		const cuComplex *A, const int *lda, const cuComplex *B,
		const int *ldb, const cuComplex *beta, cuComplex *C, const int *ldc,
		int is_splitA, float split);
#endif

#if defined(__PHIGEMM_PROFILE)
void PHIGEMM_M (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const cuComplex *alpha,
		const cuComplex *A, const int *lda, const cuComplex *B,
		const int *ldb, const cuComplex *beta, cuComplex *C, const int *ldc,
		const char *file, const char * line)
#else
void PHIGEMM_M (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const cuComplex *alpha,
		const cuComplex *A, const int *lda, const cuComplex *B,
		const int *ldb, const cuComplex *beta, cuComplex *C, const int *ldc)
#endif
{
	double time_call;
	int tmp, p1, p2;
	int a_offset, b_offset, c_offset;
	size_t memsize_gpu, mem_gpu;
	float split = -1;
	static int ground_level = 1;
	static int splitting_steps;
	int first_call = 0;
	int local_init = 0;

	/* determine which matrix is to be splitted */
	int is_splitA = -1;

#if defined(__PHIGEMM_PROFILE)
	double start, stop;
#endif

#if defined(__PHIGEMM_PROFILE)
	if ( ground_level) {
		first_call = 1;
		splitting_steps = 0;
		start = phigemm_cclock();
	}
#endif


#if defined(__PHIGEMM_CPUONLY)
	gemm_mkl(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta,C, ldc);
#else

	/* if the input matrix if pretty small, we will perform the computation on CPU */
	if ( (*n) < 64 || (*m) < 64 || (*k) < 64 )
	{
		gemm_mkl(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta,C, ldc);
		return;
	}

	if ( ground_level && !phiGemmIsInit()  )
	{
		fprintf(stderr, "*** phiGEMM *** ERROR *** Missing initialization. Do self-init.\n"); fflush(stdout);
		exit(-1);
	}

	is_splitA = (*n > *m) ? 0:1;

	/* Assign the split factor for phidgemm (2: CGEMM) */
	split = myPhiGemmTng.split[2];

	/* recursive splitting */
	/* There is an assumption here: all the cards has the same amount of memory.
	 * This can be not true at all! */
	memsize_gpu = myPhiGemmHdl.smem[0] * myPhiGemmEnv.numDevices;

	if ( is_splitA )
	{

		mem_gpu = memOccupancy(is_splitA, split, *m, *n, *k) * sizeof(cuComplex);

		if ( mem_gpu * myPhiGemmEnv.numDevices > memsize_gpu )
		{
			splitting_steps++;
			ground_level = 0;

#if defined(__PHIGEMM_DEBUG)
			printf("*** phiGEMM *** Dimensions\t%d\t%d\t%d\t( %lu bytes) too big to fit the GPU memory (%lu bytes), split A(%d, %d)...\n",
					*m, *n, *k, (unsigned long)mem_gpu, (unsigned long)memsize_gpu, *m, *n);  fflush(stdout);
#endif

			bestFit(is_splitA, split, *m, *n, *k, sizeof(cuComplex), &p1, &p2);

			a_offset = ( *transa == 'n' || *transa == 'N' )? p1 : ((*lda)*p1);
			c_offset = p1;

#if defined(__PHIGEMM_PROFILE)
			PHIGEMM_M(transa, transb, &p1, n, k, alpha, A, lda, B, ldb, beta, C, ldc, file, line);
			PHIGEMM_M(transa, transb, &p2, n, k, alpha, A + a_offset, lda, B, ldb, beta, C + c_offset, ldc, file, line);
#else
			PHIGEMM_M(transa, transb, &p1, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
			PHIGEMM_M(transa, transb, &p2, n, k, alpha, A + a_offset, lda, B, ldb, beta, C + c_offset, ldc);
#endif
		} else {

#if defined(__PHIGEMM_PROFILE)
			PHIGEMM_CGEMM_MF(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, is_splitA, split, file, line);
#else
			PHIGEMM_CGEMM_MF(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, is_splitA, split);
#endif
		}

	} else {

		mem_gpu = memOccupancy(is_splitA, split, *m, *n, *k) * sizeof(cuComplex);

		if ( mem_gpu * myPhiGemmEnv.numDevices > memsize_gpu )
		{
			ground_level = 0;
			splitting_steps++;

#if defined(__PHIGEMM_DEBUG)
			printf("*** phiGEMM *** Dimensions\t%d\t%d\t%d\t( %lu bytes) too big to fit the GPU memory (%lu bytes), split B( %d, %d )...\n",
					*m, *n, *k, (unsigned long)mem_gpu, (unsigned long)memsize_gpu, *k, *n); fflush(stdout);
#endif

			bestFit(is_splitA, split, *m, *n, *k, sizeof(cuComplex), &p1, &p2);

			b_offset = ( *transb == 'n' || *transb == 'N' )? ((*ldb)*p1) : p1;
			c_offset = (*ldc)*p1;

#if defined(__PHIGEMM_PROFILE)
			PHIGEMM_M(transa, transb, m, &p1, k, alpha, A, lda, B, ldb, beta, C, ldc, file, line);
			PHIGEMM_M(transa, transb, m, &p2, k, alpha, A, lda, B + b_offset, ldb, beta, C + c_offset, ldc, file, line);
#else
			PHIGEMM_M(transa, transb, m, &p1, k, alpha, A, lda, B, ldb, beta, C, ldc);
			PHIGEMM_M(transa, transb, m, &p2, k, alpha, A, lda, B + b_offset, ldb, beta, C + c_offset, ldc);
#endif
		} else {

#if defined(__PHIGEMM_PROFILE)
			PHIGEMM_CGEMM_MF(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, is_splitA, split, file, line);
#else
			PHIGEMM_CGEMM_MF(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, is_splitA, split);
#endif
		}
	}

	if ( cudaSetDevice(myPhiGemmHdl.devId[0]) != cudaSuccess) {
		printf("*** phiGEMM *** ERROR *** cudaSetDevice failed!\n");
		exit(EXIT_FAILURE);
	}
#endif

#if defined(__PHIGEMM_PROFILE)
	if ( first_call) {
		ground_level = 1;
		first_call = 0;
		stop = phigemm_cclock() - start;
		/* Comma-Separated Value (csv) format:
		 * file, line, nGPU, nThreads, transA, transB, m, n, k, spliting_steps, split_factor, time, GFlops */
		fprintf (myPhiGemmEnv.profileFile, "%s, %s, %d, %d, %c, %c, %d, %d, %d, %d, %.3f, %10.6f, %10.4f\n", file, line, myPhiGemmEnv.numDevices, myPhiGemmEnv.cores, *transa, *transb, *m, *n, *k, splitting_steps, split, stop, 1.e-6 * PHIGEMM_FLOPS( (double)(*m), (double)(*n), (double)(*k) )/(stop*1000));
	}
#endif

	return;
}

#if defined(__PHIGEMM_PROFILE)
void PHIGEMM_CGEMM_MF (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const cuComplex *alpha,
		const cuComplex *A, const int *lda, const cuComplex *B,
		const int *ldb, const cuComplex *beta, cuComplex *C, const int *ldc,
		int is_splitA, float split,
		const char *file, const char * line)
#else
void PHIGEMM_CGEMM_MF (const char *transa, const char *transb, const int *m,
		const int *n, const int *k, const cuComplex *alpha,
		const cuComplex *A, const int *lda, const cuComplex *B,
		const int *ldb, const cuComplex *beta, cuComplex *C, const int *ldc,
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

	size_t shift = 0;
	void *devPtrA[NSTREAMS *MAX_GPUS], *devPtrB[NSTREAMS *MAX_GPUS], *devPtrC[NSTREAMS *MAX_GPUS];
	cublasStatus_t status;
	cudaError_t cudaErr;

#if defined(__PHIGEMM_DEBUG)
	/* timing using CUDA events */
	cudaEvent_t events[myPhiGemmEnv.numDevices * NSTREAMS][7];

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
		// if (*m > 128) tmp = floor(tmp/64.0)*64;
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

		shift = 0;
		devPtrA[iDev]=(char *) myPhiGemmHdl.pmem[iDev] + shift;

#if defined(__PHIGEMM_DEBUG)
		for (j = 0; j < 7; j++)
			cudaEventCreate(&(events[iDev % myPhiGemmEnv.numDevices][j]));

		cudaEventRecord(events[iDev][0], myPhiGemmHdl.stream[iDev] );
#endif

		if ( is_transa ) {
			status = cublasSetMatrixAsync (k_h2d[iDev], m_h2d[iDev],
					sizeof(cuComplex), A+shiftA, *lda, devPtrA[iDev],
					k_gpu[iDev], myPhiGemmHdl.stream[iDev]);
			shiftA += m_h2d[iDev] * (*lda);
		} else {
			status = cublasSetMatrixAsync (m_h2d[iDev], k_h2d[iDev],
					sizeof(cuComplex), A+shiftA, *lda, devPtrA[iDev],
					m_gpu[iDev], myPhiGemmHdl.stream[iDev]);
			shiftA += m_h2d[iDev];
		}

#if defined(__PHIGEMM_DEBUG)
		cudaEventRecord(events[iDev][1], myPhiGemmHdl.stream[iDev] );
#endif

		shift += (EVENIZE(m_gpu[iDev] * k_gpu[iDev])) *sizeof(cuComplex);
		devPtrB[iDev] = (char *) myPhiGemmHdl.pmem[iDev] + shift;

		if ( is_transb ) {
			status = cublasSetMatrixAsync (n_h2d[iDev], k_h2d[iDev],
					sizeof(cuComplex), B+shiftB, *ldb, devPtrB[iDev],
					n_gpu[iDev], myPhiGemmHdl.stream[iDev]);
			shiftB += n_h2d[iDev];
		} else {
			status = cublasSetMatrixAsync (k_h2d[iDev], n_h2d[iDev],
					sizeof(cuComplex), B+shiftB, *ldb, devPtrB[iDev],
					k_gpu[iDev], myPhiGemmHdl.stream[iDev]);
			shiftB += n_h2d[iDev] * (*ldb);
		}

#if defined(__PHIGEMM_DEBUG)
		cudaEventRecord(events[iDev][2], myPhiGemmHdl.stream[iDev] );
#endif

		/* set the matrix C to device */
		shift += (EVENIZE(k_gpu[iDev] * n_gpu[iDev]) )*sizeof(cuComplex);
		devPtrC[iDev] = (char *) myPhiGemmHdl.pmem[iDev] + shift;

		if ( beta->x != 0.0 || beta->y != 0.0 ){
			status = cublasSetMatrixAsync (m_h2d[iDev], n_h2d[iDev],
					sizeof(cuComplex), C+shiftC, *ldc, devPtrC[iDev],
					m_gpu[iDev], myPhiGemmHdl.stream[iDev]);
		}

#if defined(__PHIGEMM_DEBUG)
		cudaEventRecord(events[iDev][3], myPhiGemmHdl.stream[iDev] );
#endif

#if defined(__PHIGEMM_PINNED) || defined(__PHIGEMM_MULTI_GPU)

		gpu_lda = m_gpu[iDev];
		gpu_ldb = k_gpu[iDev];

		if ( is_transa ) gpu_lda = k_gpu[iDev];
		if ( is_transb ) gpu_ldb = n_gpu[iDev];

		cublasGemm (myPhiGemmHdl.handle[ iDev ], cu_transa, cu_transb,
				m_gpu[iDev], n_gpu[iDev], k_gpu[iDev],
				alpha, devPtrA[iDev], gpu_lda, devPtrB[iDev], gpu_ldb,
				beta, devPtrC[iDev], m_gpu[iDev]);

#if defined(__PHIGEMM_DEBUG)
		cudaEventRecord(events[iDev][4], myPhiGemmHdl.stream[iDev] );
#endif

		status = cublasGetMatrixAsync (m_h2d[iDev], n_h2d[iDev],
				sizeof(cuComplex), devPtrC[iDev], m_gpu[iDev], C+shiftC,
				*ldc, myPhiGemmHdl.stream[iDev]);

		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf (stderr, "!!!! GPU %d: device access error (D2H C) %d\n", iDev, status); fflush(stderr);
		}

#if defined(__PHIGEMM_DEBUG)
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

#else

		gpu_lda = m_gpu[iDev];
		gpu_ldb = k_gpu[iDev];

		if ( is_transa ) gpu_lda = k_gpu[iDev];
		if ( is_transb ) gpu_ldb = n_gpu[iDev];

		cublasGemm (myPhiGemmHdl.handle[ iDev ], cu_transa, cu_transb, m_gpu[iDev],
				n_gpu[iDev], k_gpu[iDev], alpha, devPtrA[iDev],
				gpu_lda, devPtrB[iDev], gpu_ldb, beta, devPtrC[iDev],
				m_gpu[iDev]);

		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf (stderr, "!!!! GPU %d: device access error (D2H C) %d\n", iDev, status); fflush(stderr);
		}

#if defined(__PHIGEMM_DEBUG)
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

#if defined(__PHIGEMM_DEBUG)
	start_gemm_cpu = phigemm_cclock();
#endif

	gemm_mkl(transa, transb, &m_cpu, &n_cpu, &k_cpu, alpha, A+a_offset,
			lda, B+b_offset, ldb, beta, C+c_offset, ldc);

#if defined(__PHIGEMM_DEBUG)
	stop_gemm_cpu= phigemm_cclock();
#endif

	shiftC = 0;
	for (iDev = 0; iDev < myPhiGemmEnv.numDevices * NSTREAMS; iDev++) {
		cudaSetDevice(myPhiGemmHdl.devId[iDev % myPhiGemmEnv.numDevices]);

#if defined(__PHIGEMM_DEBUG)
		cudaEventRecord(events[iDev][5], myPhiGemmHdl.stream[iDev] );
#endif

		status = cublasGetMatrixAsync (m_h2d[iDev], n_h2d[iDev],
				sizeof(cuComplex), devPtrC[iDev], m_gpu[iDev], C+shiftC,
				*ldc, myPhiGemmHdl.stream[iDev]);

		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf (stderr, "!!!! GPU %d: device access error (D2H C) %d\n", iDev, status); fflush(stderr);
		}

#if defined(__PHIGEMM_DEBUG)
		cudaEventRecord(events[iDev][6], myPhiGemmHdl.stream[iDev] );
#endif

		if (is_splitA) {
			shiftB = 0;
			shiftC += m_h2d[iDev];
		} else {
			shiftA = 0;
			shiftC += n_h2d[iDev] * (*ldc);
		}

		cudaErr = (cudaError_t) cudaStreamSynchronize( myPhiGemmHdl.stream[ iDev ] );
		if (cudaErr != cudaSuccess) {
			printf ( "!!!! 4 - cudaDeviceSynchronize error (C) %d\n", cudaErr); fflush(stdout);
		}
	}
#endif

#if defined(__PHIGEMM_DEBUG)
	stop_gemm_total = phigemm_cclock();

	float time_temp, time_mem_h2d, time_gemm_cuda, time_mem_d2h;
	double time_total = stop_gemm_total - start_gemm_total;
	double time_mkl = stop_gemm_cpu - start_gemm_cpu;
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

		/* For best split, the time to asynchronously move data to device and compute the MxM should be equal
		 * to the time that CPU spent to perform its portion of the GEMM.
		 * NOTE: if (unbalance > 0) the CPU has too less work to do (and the GPU too much) -> decrease the split
		 * 		 if (unbalance < 0) the GPU has too less work to do (and the CPU too much) -> increase the split
		 * */
#if defined(__PHIGEMM_PINNED) && defined(__PHIGEMM_MULTI_GPU)
		unbalance = (time_mem_h2d + time_gemm_cuda + time_mem_d2h) - time_mkl;
#elif defined(__PHIGEMM_PINNED)
		unbalance = (time_mem_h2d + time_gemm_cuda) - time_mkl;
#else
		unbalance = time_gemm_cuda - time_mkl;
#endif

		if ( is_splitA ) {
#if defined(__PHIGEMM_PROFILE)
			printf ("[PHIGEMM_DEBUG - %s:%s - GPU %d] %d (%d %d, %5.4f) %d %d ~ H2D:%9.6fs (%6.4fGB/s) MKL:%9.6fs (%5.4fGflops) CUBLAS: %9.6fs (%7.4fGflops) D2H:%9.6fs (%6.4fGb/s) ~ BALANCE: %9.6fs ~ Total: %9.6fs (%7.4fGflops)\n",
					file, line, iDev % myPhiGemmEnv.numDevices,
					*m,
					m_gpu[iDev],
					m_cpu,
					split,
				*n,
				*k,
				time_mem_h2d,
				(k_gpu[iDev]*(m_gpu[iDev]+n_gpu[iDev])+m_gpu[iDev]*n_gpu[iDev])/time_mem_h2d/(1024*1024*1024/sizeof(cuComplex)),
				time_mkl,
				1.e-6 * PHIGEMM_FLOPS( (double)m_cpu, (double)(*n), (double)(*k) )/(time_mkl*1000),
				time_gemm_cuda,
				1.e-6 * PHIGEMM_FLOPS( (double)m_gpu[iDev], (double)(*n), (double)(*k) )/(time_gemm_cuda*1000),
				time_mem_d2h,
				m_gpu[iDev]*n_gpu[iDev]/time_mem_d2h/(1024*1024*1024/sizeof(cuComplex)),
				unbalance,
				time_total,
				1.e-6 * PHIGEMM_FLOPS( (double)(*m), (double)(*n), (double)(*k) )/(time_total*1000));
#else
			printf ("[PHIGEMM_DEBUG GPU %d] %d (%d %d, %5.4f) %d %d ~ H2D:%9.6fs (%6.4fGB/s) MKL:%9.6fs (%5.4fGflops) CUBLAS: %9.6fs (%7.4fGflops) D2H:%9.6fs (%6.4fGb/s) ~ BALANCE: %9.6fs ~ Total: %9.6fs (%7.4fGflops)\n",
				iDev % myPhiGemmEnv.numDevices,
				*m,
				m_gpu[iDev],
				m_cpu,
				split,
				*n,
				*k,
				time_mem_h2d,
				(k_gpu[iDev]*(m_gpu[iDev]+n_gpu[iDev])+m_gpu[iDev]*n_gpu[iDev])/time_mem_h2d/(1024*1024*1024/sizeof(cuComplex)),
				time_mkl,
				1.e-6 * PHIGEMM_FLOPS( (double)m_cpu, (double)(*n), (double)(*k) )/(time_mkl*1000),
				time_gemm_cuda,
				1.e-6 * PHIGEMM_FLOPS( (double)m_gpu[iDev], (double)(*n), (double)(*k) )/(time_gemm_cuda*1000),
				time_mem_d2h,
				m_gpu[iDev]*n_gpu[iDev]/time_mem_d2h/(1024*1024*1024/sizeof(cuComplex)),
				unbalance,
				time_total,
				1.e-6 * PHIGEMM_FLOPS( (double)(*m), (double)(*n), (double)(*k) )/(time_total*1000));
#endif
		} else {
#if defined(__PHIGEMM_PROFILE)
			printf ("[PHIGEMM_DEBUG - %s:%s - GPU %d] %d %d (%d %d, %5.4f) %d ~ H2D:%9.6fs (%6.4fGB/s) MKL:%9.6fs (%5.4fGflops) CUBLAS: %9.6fs (%7.4fGflops) D2H:%9.6fs (%6.4fGb/s) ~ BALANCE: %9.6fs~ Total: %9.6fs (%7.4fGflops)\n",
				file, line, iDev % myPhiGemmEnv.numDevices,
				*m,
				*n,
				n_gpu[iDev],
				n_cpu,
				split,
				*k,
				time_mem_h2d,
				(k_gpu[iDev]*(m_gpu[iDev]+n_gpu[iDev])+m_gpu[iDev]*n_gpu[iDev])/time_mem_h2d/(1024*1024*1024/sizeof(cuComplex)),
				time_mkl,
				1.e-6 * PHIGEMM_FLOPS( (double)(*m), (double)n_cpu, (double)(*k) )/(time_mkl*1000),
				time_gemm_cuda,
				1.e-6 * PHIGEMM_FLOPS( (double)(*m), (double)n_gpu[iDev], (double)(*k) )/(time_gemm_cuda*1000),
				time_mem_d2h,
				m_gpu[iDev]*n_gpu[iDev]/time_mem_d2h/(1024*1024*1024/sizeof(cuComplex)),
				unbalance,
				time_total,
				1.e-6 * PHIGEMM_FLOPS( (double)(*m), (double)(*n), (double)(*k))/(time_total*1000));
#else
			printf ("[PHIGEMM_DEBUG GPU %d] %d %d (%d %d, %5.4f) %d ~ H2D:%9.6fs (%6.4fGB/s) MKL:%9.6fs (%5.4fGflops) CUBLAS: %9.6fs (%7.4fGflops) D2H:%9.6fs (%6.4fGb/s) ~ BALANCE: %9.6fs~ Total: %9.6fs (%7.4fGflops)\n",
				iDev % myPhiGemmEnv.numDevices,
				*m,
				*n,
				n_gpu[iDev],
				n_cpu,
				split,
				*k,
				time_mem_h2d,
				(k_gpu[iDev]*(m_gpu[iDev]+n_gpu[iDev])+m_gpu[iDev]*n_gpu[iDev])/time_mem_h2d/(1024*1024*1024/sizeof(cuComplex)),
				time_mkl,
				1.e-6 * PHIGEMM_FLOPS( (double)(*m), (double)n_cpu, (double)(*k) )/(time_mkl*1000),
				time_gemm_cuda,
				1.e-6 * PHIGEMM_FLOPS( (double)(*m), (double)n_gpu[iDev], (double)(*k) )/(time_gemm_cuda*1000),
				time_mem_d2h,
				m_gpu[iDev]*n_gpu[iDev]/time_mem_d2h/(1024*1024*1024/sizeof(cuComplex)),
				unbalance,
				time_total,
				1.e-6 * PHIGEMM_FLOPS( (double)(*m), (double)(*n), (double)(*k) )/(time_total*1000));
#endif
		}
		fflush(stdout);
	}

	/* Destroy CUDA events */
	for (i = 0; i < myPhiGemmEnv.numDevices * NSTREAMS; i++) {
		cudaSetDevice(myPhiGemmHdl.devId[i % myPhiGemmEnv.numDevices]);
		for (j = 0; j < 7; j++)
			cudaEventDestroy(events[i][j]);
	}
#endif
}
