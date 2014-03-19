/*****************************************************************************\
 * Copyright (C) 2011-2014 Quantum ESPRESSO Foundation
 *
 * This file is distributed under the terms of the
 * GNU General Public License. See the file `License'
 * in the root directory of the present distribution,
 * or http://www.gnu.org/copyleft/gpl.txt .
 *
 * Filippo Spiga (filippo.spiga@quantum-espresso.org)
\*****************************************************************************/

/*
 * This is a simple auto-tuner program. It computed a sub-optimal split
 * factor as a weighted mean across three different measurements obtained
 * multiplying three different matrix shapes. The results can be used as a
 * seed for PHI_DGEMM_SPLIT and PHI_ZGEMM_SPLIT. It is supposed to run for
 * no more than 30 seconds. No sanity checks are embedded.
 *
 * Version 1.0 -- March 23, 2014
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "phigemm.h"

// Default settings
#define __FRACTION_OF_DEVICE_MEM_TO_USE__ 0.8
#define nGPU 1
#define lowerSplitFactor 0.9
#define upperSplitFactor 1.0
#define stepSplitFactor 0.025

// Global variables
typedef void* serialTestMemDevPtr[nGPU];
typedef size_t serialTestMemSizes[nGPU];
typedef int serialTestDeviceIds[nGPU];
serialTestMemDevPtr test_scratch;
serialTestMemSizes memsize;
serialTestDeviceIds devicesToBond;

typedef struct timestruct
{
	unsigned int sec;
	unsigned int usec;
} TimeStruct;

double seconds(){

	struct timeval tmp;
	double sec;
	gettimeofday( &tmp, (struct timezone *)0 );
	sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
	return sec;
}  

float autotuning(int nGPU, phiGemmMemDevPtr* dev_ptr, phiGemmMemSizes* dev_memsize, int * deviceToBond, int tag) {



	return computed_split;
}


int main(int argc, char **argv)
{  
	size_t buffer_size = 0L;

	double cpu_time, gpu_time, hybrid_time, kernel_time, d2h_time, h2d_time;
	double t1, t2, t3;
	double buf_size;
	size_t shift = 0;

	size_t freeMem, totalMem, mem_gpu;

	int phiGemmNumDevices, ierr;
	cudaError_t gpu_err;

	double *A, *B, *C, *C_mkl, *C_phigemm, *C_cuda, * buffer_memory_ptr;
	void *d_A, *d_B, *d_C;

	int m, n, k, i, j, err, is_transa[4], is_transb[4], nGPU, count;
	float lowerSplitFactor, upperSplitFactor, stepSplitFactor, currentSplitFactor;
	char transa, transb;

	unsigned int tmp_error, tmp_flags;



	test_scratch[0] = NULL;
	memsize[0] = (size_t) 0;
	devicesToBond[0]= 0; // Just an assumption...

	// Attempt to establish a runtime API context
	if ( cudaSetDevice(devicesToBond[0]) != cudaSuccess ) {
		printf( "*** ERROR cudaSetDevice( %d ) failed!",i );
		exit( EXIT_FAILURE );
	}

	memsize[0] = (size_t) 0;

	// First memory allocation
	ierr = cudaMalloc ( (void**) &(test_scratch[0]), memsize[0]  );
	if ( ierr != cudaSuccess) {
		fprintf( stderr, "\nError in (first zero) memory allocation [GPU %d] , program will be terminated!!! Bye...\n\n", i);
		exit(EXIT_FAILURE);
	}

	cudaMemGetInfo((size_t*)&freeMem, (size_t*)&totalMem);

#if defined(__DEBUG)
	printf("[GPU %d] before: %lu (total: %lu)\n", i, (unsigned long)freeMem, (unsigned long)totalMem); fflush(stdout);
#endif

	memsize[ i ] = (size_t) (freeMem * __FRACTION_OF_DEVICE_MEM_TO_USE__);

	// Proper memory allocation
	ierr = cudaMalloc ( (void**) &(test_scratch[ i ]), (size_t) memsize[ i ] );
	if ( ierr != cudaSuccess) {
		fprintf( stderr, "\nError in memory allocation [GPU %d] , program will be terminated (%d)!!! Bye...\n\n", i, ierr );
		exit(EXIT_FAILURE);
	}

#if defined(__DEBUG)
	cudaMemGetInfo((size_t*)&freeMem, (size_t*)&totalMem);
	printf("[GPU %d] after: %lu (total: %lu)\n", i, (unsigned long)freeMem, (unsigned long)totalMem); fflush(stdout);
#endif


#if defined (__DEBUG)
	// Not really necessary...
	cudaMemset( test_scratch[ 0 ], 0, memsize[ 0 ] );
#endif

	// init phiGEMM
	phiGemmInit( nGPU, (serialTestMemDevPtr*)&test_scratch, (serialTestMemSizes *)&memsize, (int *)&devicesToBond, 0);
	cudaDeviceSynchronize();

	float best_split = (nGPU, (serialTestMemDevPtr*)&test_scratch, (serialTestMemSizes *)&memsize, (int *)&devicesToBond, 0);

	return best_split;

}

	m = atoi( argv[ 2 ] );
	n = atoi( argv[ 3 ] );
	k = atoi( argv[ 4 ] );

	/* Allocating memory on the CPU ... */
	buffer_size = ( size_t ) ( ( ( ((m * k)%2==0) ? (m * k) : (m * k)+1 ) +
			( ((n * k)%2==0) ? (n * k) : (n * k)+1 ) +
			( ((m * k)%2==0) ? (m * n) : (m * n)+1 ) ) * sizeof(double ) );

#if defined(__PHITEST_MEM_PINNED)
	if( cudaHostAlloc( ( void ** ) &buffer_memory_ptr, buffer_size, cudaHostAllocPortable ) != cudaSuccess ) {
		printf( "*** ERROR allocating PINNED MEMORY on CPU\n" );
		exit( EXIT_FAILURE );
	}
#else
	if( ( buffer_memory_ptr = ( double * ) malloc( buffer_size ) ) == NULL ) {
		printf( "*** ERROR allocating MEMORY on CPU\n"  );
		exit( EXIT_FAILURE );
	}
#endif

	A = ( double * ) buffer_memory_ptr;
	B = A + (m * k);
	C_phigemm = B + (k * n);
	// memset( buffer_memory_ptr, 0, buffer_size );

#if defined(__PHITEST_MEM_PINNED) && !defined(__PHIGEMM_CPUONLY)
	if( cudaHostAlloc( ( void ** ) &C, m * n * sizeof( double ), cudaHostAllocPortable ) != cudaSuccess ) {
		printf( "*** ERROR allocating PINNED MEMORY on cpu\n" );
		exit( EXIT_FAILURE );
	}
#else
	C = ( double * ) malloc( m * n * sizeof( double ) );
#endif

	memset( C, 0, m * n * sizeof( double ) );

#if defined(__CUDA_TYPE_DOUBLE)
	double alpha=0.33, beta=-0.25;
#else defined(__CUDA_TYPE_DOUBLE_COMPLEX)
	phiDoubleComplex alpha, beta;
	phigemm_set_real_part(alpha, (double) 2.0 );
	phigemm_set_img_part(alpha,  (double) 1.0 );
	phigemm_set_real_part(beta,  (double) 1.0 );
	phigemm_set_img_part(beta,  (double) -0.5 );
#endif

	for ( j = 0; j < m; j++ ) {
		srand ( time(NULL) );
		for ( i = 0; i < k; i++ ) {
			int index = i * m + j;
#if defined(__CUDA_TYPE_DOUBLE_COMPLEX)
			phigemm_set_real_part( A[ index ], ( SUBdouble ) rand()/(RAND_MAX+1.0) );
			phigemm_set_img_part( A[ index ], ( SUBdouble ) rand()/(RAND_MAX+1.0) );
#else
			A[ index ] =  ( double )  rand()/(RAND_MAX+1.0);
#endif
		}
	}

	for ( j = 0; j < k; j++ ) {
		srand ( time(NULL) );
		for ( i = 0; i < n; i++ ) {
			int index = i * k + j;
#if defined(__CUDA_TYPE_DOUBLE_COMPLEX)
			phigemm_set_real_part( B[ index ], ( SUBdouble ) rand()/(RAND_MAX+1.0) );
			phigemm_set_img_part( B[ index ], ( SUBdouble ) rand()/(RAND_MAX+1.0) );
#else
			B[ index ] =  ( double )  rand()/(RAND_MAX+1.0);
#endif
		}
	}


	for ( j = 0; j < m; j++ ) {
		srand ( time(NULL) );
		for ( i = 0; i < n; i++ ) {
			int index = i * m + j;
#if defined(__CUDA_TYPE_DOUBLE_COMPLEX)
			phigemm_set_real_part( C[ index ], ( SUBdouble ) rand()/(RAND_MAX+1.0) );
			phigemm_set_img_part( C[ index ], ( SUBdouble ) rand()/(RAND_MAX+1.0) );
#else
			C[ index ] =  ( double )  rand()/(RAND_MAX+1.0);
#endif
		}
	}


	transa[0] = 'n'; transb[0] = 'n';

#if defined(__CUDA_TYPE_DOUBLE_COMPLEX)
	transa[1] = 'c';
#else
	transa[1] = 't';
#endif
	transb[1] = 'n';

	transa[2] = 'n';
#if defined(__CUDA_TYPE_DOUBLE_COMPLEX)
	transb[2] = 'c';
#else
	transb[2] = 't';
#endif

#if defined(__CUDA_TYPE_DOUBLE_COMPLEX)
	transa[3] = 'c'; transb[3] = 'c';
#else
	transa[3] = 't'; transb[3] = 't';
#endif

	for( count = 0; count < 4; count +=1 ){
		is_transa[count] = (transa[count] != 'n') ? 1 : 0;
		is_transb[count] = (transb[count] != 'n') ? 1 : 0;
	}

	for( count = 0; count < 1; count +=1 ){

		int lda = m;
		int ldb = k;

		if ( is_transa[ count ] ) lda = k;
		if ( is_transb[ count ] ) ldb = n;

		/* ----------------------- run MxM using MKL ---------------------- */
		C_mkl = ( double* ) malloc( m * n * sizeof( double ) );

#if 0
		// Fake call to avoid caching effects.... -- BAD IF TEST IS LARGE
		memset( C_mkl, (double)-1.0, m * n * sizeof( double ) );
		dgemm_(&transa[ count ], &transb[ count ], &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C_mkl, &m);
#endif
		memset( C_mkl, 0, m * n * sizeof( double ) );
		for ( j = 0; j < m; j++ ) {
			for ( i = 0; i < n; i++ ) {
				int index = i * m + j;
#if defined(__CUDA_TYPE_DOUBLE_COMPLEX)
			phigemm_set_real_part( C_mkl[ index ], phigemm_get_real_part(C[ index ]) );
			phigemm_set_img_part( C_mkl[ index ], phigemm_get_img_part(C[ index ]) );
#else
				C_mkl[ index ] = C[ index ];
#endif
			}
		}

		t1 = seconds();
		dgemm_(&transa[ count ], &transb[ count ], &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C_mkl, &m);
		cpu_time = seconds() - t1;

		fprintf( stdout, "\nMKL ( %d cores ) GEMM: Elapsed time = %10.6f s - RPeak = %10.4f GFlop/s\n", atoi( getenv( "OMP_NUM_THREADS" ) ), cpu_time, ( 1.e-6 ) * PHIGEMM_FLOPS(( double ) m, ( double ) n, ( double ) k) / (cpu_time*1000) );
		fflush( stdout );

		/* ----------------------------------------------------------- */

		/* --------------------- test the CUBLAS --------------------- */
#if !defined(__PHIGEMM_CPUONLY)

		mem_gpu = ( m*k + k*n + m*n ) * sizeof(double);

		if (mem_gpu > memsize[ 0 ] ) {
			/* I simply cannot run this GEMM on one single GPU ... */
			gpu_time = 0;
		} else {
			cublasHandle_t handle;

			/* Attempt to establish a runtime API context */
			if ( cudaSetDevice( devicesToBond[0] ) != cudaSuccess ) {
				printf( "*** ERROR cudaSetDevice( %d ) failed!",i );
				exit( EXIT_FAILURE );
			}

#if defined(__PHITEST_MEM_PINNED)
			if( cudaHostAlloc( ( void ** ) &C_cuda, m * n * sizeof( double ), cudaHostAllocPortable ) != cudaSuccess ) {
				printf( "*** ERROR allocating PINNED MEMORY on cpu\n" );
				exit( EXIT_FAILURE );
			}
#else
			C_cuda = ( double* ) malloc( m * n * sizeof( double ) );
#endif

			memset( C_cuda, 0, m * n * sizeof( double ) );

			for ( j = 0; j < m; j++ ) {
				for ( i = 0; i < n; i++ ) {
					int index = i * m + j;
#if defined(__CUDA_TYPE_DOUBLE_COMPLEX)
					phigemm_set_real_part( C_cuda[ index ], phigemm_get_real_part(C[ index ]) );
					phigemm_set_img_part( C_cuda[ index ], phigemm_get_img_part(C[ index ]) );
#else
					C_cuda[ index ] = C[ index ];
#endif
				}
			}

			if ( cudaSetDevice(devicesToBond[0]) != cudaSuccess) {
				printf("*** ERROR cudaSetDevice\n");
				exit(EXIT_FAILURE);
			}

			if ( cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS ) {
				printf ("CUBLAS initialization failed\n");
				return EXIT_FAILURE;
			}

			// Be AWARE when you load data in this way... always align by 16!
			shift = 0;
			d_A[0] = (char*) test_scratch[0] + shift;
			shift += ( ((m * k)%2==0) ? (m * k) : (m * k)+1 )*sizeof(double);
			d_B[0] = (char*) test_scratch[0] + shift;
			shift += ( ((k * n)%2==0) ? (k * n) : (k * n)+1 )*sizeof(double);
			d_C[0] = (char*) test_scratch[0] + shift;

			cublasOperation_t cu_transa, cu_transb;
			cu_transa =  ( (transa[ count ] == 'c') || (transa[ count ] == 'C') ) ? CUBLAS_OP_C : CUBLAS_OP_N;
			cu_transa =  ( (transa[ count ] == 't') || (transa[ count ] == 'T') ) ? CUBLAS_OP_T : cu_transa;
			cu_transa =  ( (transa[ count ] == 'n') || (transa[ count ] == 'N') ) ? CUBLAS_OP_N : cu_transa;
			cu_transb =  ( (transb[ count ] == 'c') || (transb[ count ] == 'C') ) ? CUBLAS_OP_C : CUBLAS_OP_N;
			cu_transb =  ( (transb[ count ] == 't') || (transb[ count ] == 'T') ) ? CUBLAS_OP_T : cu_transb;
			cu_transb =  ( (transb[ count ] == 'n') || (transb[ count ] == 'N') ) ? CUBLAS_OP_N : cu_transb;

			t1 = seconds();

			if ( is_transa[count] )
				cublasSetMatrix(k, m, sizeof(double), A, lda, (double*) d_A[0], lda);
			else
				cublasSetMatrix(m, k, sizeof(double), A, lda, (double*) d_A[0], lda);

			if ( is_transb[count] )
				cublasSetMatrix(n, k, sizeof(double), B, ldb, (double*) d_B[0], ldb);
			else
				cublasSetMatrix(k, n, sizeof(double), B, ldb, (double*) d_B[0], ldb);

			cublasSetMatrix(m, n, sizeof(double), C_cuda, m, (double*) d_C[0], m);

			h2d_time = seconds() - t1;

			t2 = seconds();
			cublasDgemm(handle, cu_transa, cu_transb, m, n, k, &alpha, d_A[0], lda, d_B[0], ldb, &beta, d_C[0], m);
			cudaDeviceSynchronize();
			kernel_time = seconds() - t2;

			t3 = seconds();
			cublasGetMatrix(m, n, sizeof(double), (double*) d_C[0], m, C_cuda, m);

			/* gpu_time =  H2D + COMPUTATION + D2H */
			gpu_time = seconds() - t1;
			d2h_time = seconds() - t3;

			fprintf( stdout, "CUBLAS (kernel + transfer)\t: Elapsed time = %10.6f s - RPeak = %10.4f GFlop/s (H2D: %9.6fs, D2H: %9.6fs)\n",  gpu_time, ( 1.e-6 ) * PHIGEMM_FLOPS(( double ) m, ( double ) n, ( double ) k) / (gpu_time*1000), h2d_time, d2h_time );
			fprintf( stdout, "CUBLAS (only kernel)\t\t: Elapsed time = %10.6f s - RPeak = %10.4f GFlop/s\n\n",  kernel_time, ( 1.e-6 ) * PHIGEMM_FLOPS(( double ) m, ( double ) n, ( double ) k) / (kernel_time*1000) );
			fflush( stdout );

		}
#else
		// Only because output has to be "human readable"
		fprintf( stdout, "\n");fflush( stdout );
#endif
		/* ----------------------------------------------------------- */

		/* --------------------- Run MxM using PHIGEMM -------------------- */
		currentSplitFactor = lowerSplitFactor;
		do {
			for ( j = 0; j < m; j++ ) {
				for ( i = 0; i < n; i++ ) {
					int index = i * m + j;
#if defined(__CUDA_TYPE_DOUBLE_COMPLEX)
					phigemm_set_real_part( C_phigemm[ index ], phigemm_get_real_part(C[ index ]) );
					phigemm_set_img_part( C_phigemm[ index ], phigemm_get_img_part(C[ index ]) );
#else
					C_phigemm[ index ] = C[ index ];
#endif
				}
			}

#if !defined(__PHIGEMM_CPUONLY)
			/* Optimal.... but probably not optimal anymore! */
			// currentSplitFactor = (( 2.e-9 ) * ( double ) m * ( double ) n * ( double ) k / kernel_time)*nGPU / ((( 2.e-9 ) * ( double ) m * ( double ) n * ( double ) k / kernel_time)*nGPU + (( 2.e-9 ) * ( double ) m * ( double ) n * ( double ) k / cpu_time) );
			float splits[4];
			splits[0] = currentSplitFactor;
			splits[1] = currentSplitFactor;
			splits[2] = currentSplitFactor;
			splits[3] = currentSplitFactor;
			phigemmSetSplitFactor((float *)&splits);
#endif

			t1 = seconds();
#if defined(__PHIGEMM_PROFILE)
			PHIGEMM_CALL(&transa[ count ], &transb[ count ], &m, &n, &k,
					(const double *) &alpha, (const double *) A, &lda,
					(const double *) B, &ldb, (const double *) &beta,
					C_phigemm, &m, __FILE__, __LINESTR__);
#else
			PHIGEMM_CALL(&transa[ count ], &transb[ count ], &m, &n, &k,
					(const double *) &alpha, (const double *)A, &lda,
					(const double *) B, &ldb, (const double *) &beta,
					C_phigemm, &m);
#endif
			hybrid_time = seconds() - t1;

			fprintf( stdout, "[%c%c]  phiGEMM ( %d CPU / %d GPUs ) phiGEMM (split: %5.4f): Elapsed time = %10.6f s - RPeak = %10.4f GFlop/s\t(Split = %.3f)\t errors: %c\n", transa[ count ], transb[ count ], atoi( getenv( "OMP_NUM_THREADS" ) ), nGPU, currentSplitFactor, hybrid_time, ( 1.e-6 ) * PHIGEMM_FLOPS(( double ) m, ( double ) n, ( double ) k) / (hybrid_time*1000), currentSplitFactor, (errors > 0 ? 'Y' : (errors == 0 ? 'N' : 'X')) );
			fflush( stdout );

		} while((currentSplitFactor += stepSplitFactor) <= upperSplitFactor);
		/* ----------------------------------------------------------- */

		/* end */
	}


	// Cleaning...
	phiGemmShutdown();

	if( cudaSetDevice( devicesToBond[0] ) != cudaSuccess )
	{
		fprintf( stderr, "*** ERROR cudaSetDevice\n");
		exit( EXIT_FAILURE );
	}

	if( cudaFree( test_scratch[ 0 ] ) != cudaSuccess )
	{
		fprintf( stderr, "[device:%d] Error cudaFree.\n", i);
		exit( EXIT_FAILURE );
	}

#if defined(__PHITEST_MEM_PINNED)
	cudaFreeHost( buffer_memory_ptr );
	if ( !(mem_gpu > memsize[ 0 ]) ) cudaFreeHost( C_cuda );
#else
	free( buffer_memory_ptr );
	if ( !(mem_gpu > memsize[ 0 ]) ) free( C_cuda );
#endif

	free( C_mkl );

	return 0;
}
