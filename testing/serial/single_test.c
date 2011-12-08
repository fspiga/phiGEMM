/*
 * Copyright (C) 2010-2011 Irish Centre for High-End Computing (ICHEC)
 *
 * This file is distributed under the terms of the
 * GNU General Public License. See the file `License'
 * in the root directory of the present distribution,
 * or http://www.gnu.org/copyleft/gpl.txt .
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"

#include "phigemm.h"

#include <sys/time.h>
#include <assert.h>


// Flops formula
#define GEMM_ADD(m, n, k) ((m) * (n) * (k))
#define GEMM_MUL(m, n, k) ((m) * (n) * (k))
#if defined(__CUDA_TYPE_DOUBLE_COMPLEX) || defined(__CUDA_TYPE_COMPLEX)
#define PHIGEMM_FLOPS(m, n, k) ( 6. * GEMM_MUL(m, n, k) + 2. * GEMM_ADD(m, n, k))
#else
#define PHIGEMM_FLOPS(m, n, k) (      GEMM_MUL(m, n, k) +      GEMM_ADD(m, n, k))
#endif

#define _STRING_LINE_(s) #s
#define _STRING_LINE2_(s) _STRING_LINE_(s)
#define __LINESTR__ _STRING_LINE2_(__LINE__)

/* CHECK_ERROR has problems with FLOAT... why? */
#if defined __CUDA_TYPE_FLOAT
#define MAX_ERROR 0.0000001
#else
#define MAX_ERROR 0.001
#endif

/**
 * SGEMM definitions
 */
#if defined __CUDA_TYPE_FLOAT
#define XTYPE float
#define MKL_CALL sgemm_
#define PHIGEMM_CALL phisgemm_
#define CUBLAS_GEMM cublasSgemm

/**
 * DGEMM definitions
 */
#elif defined __CUDA_TYPE_DOUBLE
#define XTYPE double
#define MKL_CALL dgemm_
#define PHIGEMM_CALL phidgemm_
#define CUBLAS_GEMM cublasDgemm

/*
 * vim M definitions
 */
#elif defined __CUDA_TYPE_COMPLEX
#define XTYPE cuComplex
#define MKL_CALL cgemm_
#define PHIGEMM_CALL phicgemm_
#define CUBLAS_GEMM cublasCgemm

/**
 * ZGEMM definitions
 */
#elif defined __CUDA_TYPE_DOUBLE_COMPLEX
#define MKL_CALL zgemm_
#define PHIGEMM_CALL phizgemm_
#define CUBLAS_GEMM cublasZgemm
#define XTYPE cuDoubleComplex

#else // default
#error a type must be defined
#endif


#define __FRACTION_OF_DEVICE_MEM_TO_USE__ 0.8

#define MAX_GPU_SERIAL_TEST 8

typedef void* serialTestMemDevPtr[MAX_GPU_SERIAL_TEST];
typedef size_t serialTestMemSizes[MAX_GPU_SERIAL_TEST];
typedef int serialTestDeviceIds[MAX_GPU_SERIAL_TEST];

serialTestMemDevPtr test_scratch;
serialTestMemSizes memsize;
serialTestDeviceIds devicesToBond;

typedef struct timestruct
{
	unsigned int sec;
	unsigned int usec;
} TimeStruct;

void swap( int a, int b){

	int tmp = a;
	a = b;
	b = tmp;
}

double seconds(){

	struct timeval tmp;
	double sec;
	gettimeofday( &tmp, (struct timezone *)0 );
	sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
	return sec;
}  

int main(int argc, char **argv)
{  
	size_t byte_GPU_buffer = 0L;

	double cpu_time, gpu_time, hybrid_time, kernel_time, d2h_time, h2d_time;
	double t1, t2, t3;
	XTYPE buf_size;

	size_t freeMem, totalMem, mem_gpu;

	int phiGemmNumDevices, ierr;

	cudaError_t gpu_err;

	XTYPE *A, *B, *C, *C_mkl, *C_phigemm, *C_cuda, * GPU_buffer_memory_ptr;
	XTYPE *d_A[MAX_GPUS], *d_B[MAX_GPUS], *d_C[MAX_GPUS];

	int m, n, k, i, j, err, is_transa[4], is_transb[4], nGPU, count;
	float lowerSplitFactor, upperSplitFactor, stepSplitFactor, currentSplitFactor;
	char transa[4], transb[4];

	unsigned int tmp_error, tmp_flags;

	if( argc != 8 )
	{
		fprintf( stderr, "\nLaunch ERROR: Use ${Executable} <nGPU> <m> <n> <k> <lower split-factor> <upper split-factor> <step>\nfor matrix multiplication C( m, n ) = A( m, k ) x B( k, n )\n" );
		exit(EXIT_FAILURE );
	}

	nGPU = atoi( argv[ 1 ] );
	m = atoi( argv[ 2 ] );
	n = atoi( argv[ 3 ] );
	k = atoi( argv[ 4 ] );
	lowerSplitFactor = atof(argv[ 5 ]);
	upperSplitFactor = atof(argv[ 6 ]);
	stepSplitFactor = atof(argv[ 7 ]);

	cudaGetDeviceCount( &phiGemmNumDevices );
	if(  nGPU < 1 || nGPU > phiGemmNumDevices )
	{
		fprintf( stderr, "\nLaunch ERROR: The number of nGPU needs to be within the [ 1, %d ] interval.", phiGemmNumDevices  );
		exit( EXIT_FAILURE);
	}

	phiGemmNumDevices = nGPU;

	for ( i = 0 ; i < nGPU ; i += 1 )
	{
		test_scratch[ i ] = NULL;
		memsize[ i ] =  0;
	}

#if defined __PERFORM_PHIGEMM_INIT
	/* GPU environment initialization */
	for ( i = 0; i < nGPU; i++ ) {

		/* bound the device */
		devicesToBond[i] = i;

		/* Attempt to establish a runtime API context */
		if ( cudaSetDevice( devicesToBond[i] ) != cudaSuccess ) {
			printf( "*** ERROR cudaSetDevice( %d ) failed!",i );
			exit( EXIT_FAILURE );
		}

		memsize[ i ] = (size_t) 0;

		ierr = cudaMalloc ( (void**) &(test_scratch[ i ]), memsize[ i ] );
		if ( ierr != cudaSuccess) {
			fprintf( stderr, "\nError in (first zero) memory allocation [GPU %d] , program will be terminated!!! Bye...\n\n", i);
			exit(EXIT_FAILURE);
		}

		cuMemGetInfo(&freeMem, &totalMem);

#if defined __CUDA_DEBUG
		printf("[GPU %d] before: %lu (total: %lu)\n", i, (unsigned long)freeMem, (unsigned long)totalMem); fflush(stdout);
#endif

		memsize[ i ] = (size_t) (freeMem * __FRACTION_OF_DEVICE_MEM_TO_USE__);

		ierr = cudaMalloc ( (void**) &(test_scratch[ i ]), (size_t) memsize[ i ] );
		if ( ierr != cudaSuccess) {
			fprintf( stderr, "\nError in memory allocation [GPU %d] , program will be terminated (%d)!!! Bye...\n\n", i, ierr );
			exit(EXIT_FAILURE);
		}

#if defined __CUDA_DEBUG
		cuMemGetInfo(&freeMem, &totalMem);
		printf("[GPU %d] after: %lu (total: %lu)\n", i, (unsigned long)freeMem, (unsigned long)totalMem); fflush(stdout);
#endif

		cudaMemset( test_scratch[ i ], 0, memsize[ i ] );
	}
#endif



#if defined __PERFORM_PHIGEMM_INIT
	phiGemmInit( nGPU, (serialTestMemDevPtr*)&test_scratch, (serialTestMemSizes *)&memsize, (int *)&devicesToBond);
#endif
	cudaDeviceSynchronize();

	/* Allocating memory on the CPU ... */
	byte_GPU_buffer = ( size_t ) ( ( m * k + k * n + m * n ) * sizeof(XTYPE ) );

#ifdef __PHITEST_MEM_PINNED
	if( cudaHostAlloc( ( void ** ) &GPU_buffer_memory_ptr, byte_GPU_buffer, cudaHostAllocPortable ) != cudaSuccess )
	{
		printf( "*** ERROR allocating PINNED MEMORY on CPU\n" );
		exit( EXIT_FAILURE );
	}
#else
	if( ( GPU_buffer_memory_ptr = ( XTYPE * ) malloc( byte_GPU_buffer ) ) == NULL )
	{
		printf( "*** ERROR allocating MEMORY on CPU\n"  );
		exit( EXIT_FAILURE );
	}

#if defined __PHITEST_FORCE_PINNED

	    /* the first call makes no sense */
//        tmp_error = (int) cuMemHostGetFlags(&tmp_flags, GPU_buffer_memory_ptr);
//        printf("[cuMemHostGetFlags] tmp_error=%d, tmp_flags=%d\n",tmp_error, tmp_flags); fflush(stdout);

        tmp_error = (int) cudaHostRegister(GPU_buffer_memory_ptr, byte_GPU_buffer, CU_MEMHOSTALLOC_PORTABLE);
        printf("[cuMemHostRegister] tmp_error=%d\n", tmp_error); fflush(stdout);

        tmp_error = (int) cudaHostGetFlags(&tmp_flags, GPU_buffer_memory_ptr);
        printf("[cuMemHostGetFlags] tmp_error=%d, tmp_flags=%d\n",tmp_error, tmp_flags); fflush(stdout);
#endif
#endif

	fprintf( stdout, "\nsizeof(XTYPE) = %lu", (size_t) sizeof(XTYPE) );
#if defined __CUDA_TYPE_FLOAT
	fprintf( stdout, "\nPERFORMING SGEMM operations\n");
#elif defined __CUDA_TYPE_COMPLEX
	fprintf( stdout, "\nPERFORMING CGEMM operations\n");
#elif defined __CUDA_TYPE_DOUBLE
	fprintf( stdout, "\nPERFORMING DGEMM operations\n");
#elif defined __CUDA_TYPE_DOUBLE_COMPLEX
	fprintf( stdout, "\nPERFORMING ZGEMM operations\n");
#endif
	// initialize host memory pointers
	A = ( XTYPE* ) GPU_buffer_memory_ptr;
	B = A + (m * k);
	C_phigemm = B + (k * n);
	memset( GPU_buffer_memory_ptr, 0, byte_GPU_buffer );

#ifdef __PHITEST_MEM_PINNED
	if( cudaHostAlloc( ( void ** ) &C, m * n * sizeof( XTYPE ), cudaHostAllocPortable ) != cudaSuccess )
	{
		printf( "*** ERROR allocating PINNED MEMORY on cpu\n" );
		exit( EXIT_FAILURE );
	}
#else
	C = ( XTYPE* ) malloc( m * n * sizeof( XTYPE ) );
#endif

	memset( C, 0, m * n * sizeof( XTYPE ) );


	/*
	 * Matrix initialization [1 .. MAX = 10]
	 * NOTE: Matrix are initialized in Column-Major order. CUBLAS library makes use of 1-based
	 *       indexing and Fortran-style column-major storage for multidimensional data to simplify
	 *       interfacing to Fortran applications. For more info see section
	 *       "Appendix B: CUBLAS Fortran Bindings" of CUBLAS library documentation.
	 */

#if (defined __CUDA_TYPE_FLOAT)
	float alpha=0.5, beta=0.15;
#elif (defined __CUDA_TYPE_DOUBLE)
	double alpha=1.33, beta=-0.25;
#elif (defined __CUDA_TYPE_COMPLEX)
	cuComplex alpha, beta;
	alpha.x = 2.0;
	alpha.y = 1.0;
	beta.x = 1.0;
	beta.y = -0.5;
#elif (defined __CUDA_TYPE_DOUBLE_COMPLEX)
	cuDoubleComplex alpha, beta;
	alpha.x = 2.0;
	alpha.y = 1.0;
	beta.x = 1.0;
	beta.y = -0.5;
#endif

	for ( j = 0; j < m; j++ ) {
		for ( i = 0; i < k; i++ ) {
			int index = i * m + j;
#if defined __CUDA_TYPE_COMPLEX
			A[ index ].x = ( float ) rand() / (RAND_MAX / 10 + 1);
			A[ index ].y = ( float ) rand() / (RAND_MAX / 10 + 1);
#elif defined __CUDA_TYPE_DOUBLE_COMPLEX
			A[ index ].x = ( double ) rand() / (RAND_MAX / 10 + 1);
			A[ index ].y = ( double ) rand() / (RAND_MAX / 10 + 1);
#else
			A[ index ] =  ( XTYPE ) rand() / (RAND_MAX / 10 + 1);
#endif
		}
	}

	for ( j = 0; j < k; j++ ) {
		for ( i = 0; i < n; i++ ) {
			int index = i * k + j;
#if defined __CUDA_TYPE_COMPLEX
			B[ index ].x = ( float ) rand() / (RAND_MAX / 10 + 1);
			B[ index ].y = ( float ) rand() / (RAND_MAX / 10 + 1);
#elif defined __CUDA_TYPE_DOUBLE_COMPLEX
			B[ index ].x = ( double ) rand() / (RAND_MAX / 10 + 1);
			B[ index ].y = ( double ) rand() / (RAND_MAX / 10 + 1);
#else
			B[ index ] =  ( XTYPE ) rand() / (RAND_MAX / 10 + 1);
#endif
		}
	}

	for ( j = 0; j < m; j++ ) {
		for ( i = 0; i < n; i++ ) {
			int index = i * m + j;
#if defined __CUDA_TYPE_COMPLEX
			C[ index ].x = ( float ) (rand() / (RAND_MAX / 10 + 1));
			C[ index ].y  = ( float ) (rand() / (RAND_MAX / 10 + 1));
#elif defined __CUDA_TYPE_DOUBLE_COMPLEX
			C[ index ].x = ( double ) (rand() / (RAND_MAX / 10 + 1));
			C[ index ].y  = ( double ) (rand() / (RAND_MAX / 10 + 1));
#else
			C[ index ] =  ( XTYPE ) (rand() / (RAND_MAX / 10 + 1));
#endif
		}
	}


	transa[0] = 'n';
	transb[0] = 'n';
	is_transa[0] = 0;
	is_transb[0] = 0;

#if defined __CUDA_TYPE_COMPLEX || defined __CUDA_TYPE_DOUBLE_COMPLEX
	transa[1] = 'c'; // 'c' for conjugate complex
#else
	transa[1] = 't';
#endif
	transb[1] = 'n';
	is_transa[1] = 1;
	is_transb[1] = 0;

	transa[2] = 'n';
#if defined __CUDA_TYPE_COMPLEX || defined __CUDA_TYPE_DOUBLE_COMPLEX
	transb[2] = 'c'; // 'c' for conjugate complex
#else
	transb[2] = 't';
#endif
	is_transa[2] = 0;
	is_transb[2] = 1;

#if defined __CUDA_TYPE_COMPLEX || defined __CUDA_TYPE_DOUBLE_COMPLEX
	transa[3] = 'c'; // 'c' for conjugate complex
	transb[3] = 'c'; // 'c' for conjugate complex
#else
	transa[3] = 't';
	transb[3] = 't';
#endif
	is_transa[3] = 1;
	is_transb[3] = 1;


	// Edit only to perform one single test ...
//	transa[0] = 'c';
//	transb[0] = 'n';
//	is_transa[0] = 1;
//	is_transb[0] = 0;

	for( count = 0; count < 1; count +=1 ){

		int lda = m;
		int ldb = k;

		if ( is_transa[ count ] ) lda = k;
		if ( is_transb[ count ] ) ldb = n;

		/* ----------------------- run MxM using MKL ---------------------- */
		C_mkl = ( XTYPE* ) malloc( m * n * sizeof( XTYPE ) );
		memset( C_mkl, 0, m * n * sizeof( XTYPE ) );

		for ( j = 0; j < m; j++ ) {
			for ( i = 0; i < n; i++ ) {
				int index = i * m + j;
#if defined __CUDA_TYPE_COMPLEX || defined __CUDA_TYPE_DOUBLE_COMPLEX
				C_mkl[ index ].x = C[ index ].x;
				C_mkl[ index ].y = C[ index ].y;
#else
				C_mkl[ index ] = C[ index ];
#endif
			}
		}


		t1 = seconds();
		MKL_CALL(&transa[ count ], &transb[ count ], &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C_mkl, &m);
		cpu_time = seconds() - t1;

		fprintf( stdout, "\nMKL ( %d cores ) GEMM: Elapsed time = %10.6f s - RPeak = %10.4f GFlop/s", atoi( getenv( "MKL_NUM_THREADS" ) ), cpu_time, ( 1.e-6 ) * PHIGEMM_FLOPS(( double ) m, ( double ) n, ( double ) k) / (cpu_time*1000) );
		fflush( stdout );

		/* ----------------------------------------------------------- */

		/* --------------------- test the CUBLAS --------------------- */

		/* if "__PERFORM_PHIGEMM_INIT" not defined, this test does not make sense
		 * because there is no memory allocated on the device. CUBLAS fails but
		 * the program can continue because we do not capture the error of
		 * cudaGetMatrix (that fails because there is no data to retrieve)
		 * */

		mem_gpu = ( m*k + k*n + m*n ) * sizeof(XTYPE);

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

#ifdef __PHITEST_MEM_PINNED
			if( cudaHostAlloc( ( void ** ) &C_cuda, m * n * sizeof( XTYPE ), cudaHostAllocPortable ) != cudaSuccess )
			{
				printf( "*** ERROR allocating PINNED MEMORY on cpu\n" );
				exit( EXIT_FAILURE );
			}
#else
			C_cuda = ( XTYPE* ) malloc( m * n * sizeof( XTYPE ) );
#endif

			memset( C_cuda, 0, m * n * sizeof( XTYPE ) );

			for ( j = 0; j < m; j++ ) {
				for ( i = 0; i < n; i++ ) {
					int index = i * m + j;
#if defined __CUDA_TYPE_COMPLEX || defined __CUDA_TYPE_DOUBLE_COMPLEX
					C_cuda[ index ].x = C[ index ].x;
					C_cuda[ index ].y = C[ index ].y;
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

			d_A[0] = (XTYPE*) test_scratch[0];
			d_B[0] = d_A[0] + (m * k);
			d_C[0] = d_B[0] + (k * n);

			cublasOperation_t cu_transa, cu_transb;
			cu_transa =  ( (transa[ count ] == 'c') || (transa[ count ] == 'C') ) ? CUBLAS_OP_C : CUBLAS_OP_N;
			cu_transa =  ( (transa[ count ] == 't') || (transa[ count ] == 'T') ) ? CUBLAS_OP_T : cu_transa;
			cu_transa =  ( (transa[ count ] == 'n') || (transa[ count ] == 'N') ) ? CUBLAS_OP_N : cu_transa;
			cu_transb =  ( (transb[ count ] == 'c') || (transb[ count ] == 'C') ) ? CUBLAS_OP_C : CUBLAS_OP_N;
			cu_transb =  ( (transb[ count ] == 't') || (transb[ count ] == 'T') ) ? CUBLAS_OP_T : cu_transb;
			cu_transb =  ( (transb[ count ] == 'n') || (transb[ count ] == 'N') ) ? CUBLAS_OP_N : cu_transb;

			t1 = seconds();

			if ( is_transa[count] )
				cublasSetMatrix(k, m, sizeof(XTYPE), A, lda, d_A[0], lda);
			else
				cublasSetMatrix(m, k, sizeof(XTYPE), A, lda, d_A[0], lda);

			if ( is_transb[count] )
				cublasSetMatrix(n, k, sizeof(XTYPE), B, ldb, d_B[0], ldb);
			else
				cublasSetMatrix(k, n, sizeof(XTYPE), B, ldb, d_B[0], ldb);

			cublasSetMatrix(m, n, sizeof(XTYPE), C_cuda, m, d_C[0], m);

			h2d_time = seconds() - t1;

			t2 = seconds();
			CUBLAS_GEMM(handle, cu_transa, cu_transb, m, n, k, &alpha, d_A[0], lda, d_B[0], ldb, &beta, d_C[0], m);
			cudaDeviceSynchronize();
			kernel_time = seconds() - t2;

			t3 = seconds();
			cublasGetMatrix(m, n, sizeof(XTYPE), d_C[0], m, C_cuda, m);

			/* gpu_time =  H2D + COMPUTATION + D2H */
			gpu_time = seconds() - t1;
			d2h_time = seconds() - t3;

			fprintf( stdout, "\nCUBLAS (kernel + transfer)\t: Elapsed time = %10.6f s - RPeak = %10.4f GFlop/s (H2D: %9.6fs, D2H: %9.6fs)\n",  gpu_time, ( 1.e-6 ) * PHIGEMM_FLOPS(( double ) m, ( double ) n, ( double ) k) / (gpu_time*1000), h2d_time, d2h_time );
			fprintf( stdout, "CUBLAS (only kernel)\t\t: Elapsed time = %10.6f s - RPeak = %10.4f GFlop/s\n\n",  kernel_time, ( 1.e-6 ) * PHIGEMM_FLOPS(( double ) m, ( double ) n, ( double ) k) / (kernel_time*1000) );
			fflush( stdout );

		}
		/* ----------------------------------------------------------- */

		/* --------------------- Run MxM using PHIGEMM -------------------- */
		currentSplitFactor = lowerSplitFactor;
		do {

			for ( j = 0; j < m; j++ ) {
				for ( i = 0; i < n; i++ ) {
					int index = i * m + j;
#if defined __CUDA_TYPE_COMPLEX || defined __CUDA_TYPE_DOUBLE_COMPLEX
					C_phigemm[ index ].x = C[ index ].x;
					C_phigemm[ index ].y = C[ index ].y;
#else
					C_phigemm[ index ] = C[ index ];
#endif
				}
			}


			/* Optimal.... but probably not optimal anymore! */
			//			currentSplitFactor = (( 2.e-9 ) * ( double ) m * ( double ) n * ( double ) k / kernel_time)*nGPU / ((( 2.e-9 ) * ( double ) m * ( double ) n * ( double ) k / kernel_time)*nGPU + (( 2.e-9 ) * ( double ) m * ( double ) n * ( double ) k / cpu_time) );
			float splits[4];
			splits[0] = currentSplitFactor;
			splits[1] = currentSplitFactor;
			splits[2] = currentSplitFactor;
			splits[3] = currentSplitFactor;
			phigemmSetSplitFactor((float *)&splits);

			t1 = seconds();
#if defined  __PHIGEMM_PROFILE
			PHIGEMM_CALL(&transa[ count ], &transb[ count ], &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C_phigemm, &m, __FILE__, __LINESTR__);
#else
			PHIGEMM_CALL(&transa[ count ], &transb[ count ], &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C_phigemm, &m);
#endif
			hybrid_time = seconds() - t1;


#ifdef __CHECK_ERROR

#if defined __CUDA_TYPE_COMPLEX || defined __CUDA_TYPE_DOUBLE_COMPLEX
			double error = 0;
			double tmp_error;

			// A easy way to check "both" ...
#pragma omp parallel for reduction (+ : error)
			for( i = 0; i < m * n ; i++ ) {

				tmp_error = (double) abs( (float)C_mkl[ i ].x - (double)C_phigemm[ i ].x );
				if (tmp_error > MAX_ERROR ) {
					//				fprintf( stdout, "\nC_mkl[ %d ].x=%15.13e  C_phigemm[ %d ].x=%15.13e", i, C_mkl[ i ].x, i, C_phigemm[ i ].x);fflush(stdout);
					//				exit(EXIT_FAILURE);
					error += tmp_error;
				}

				tmp_error = abs( (double)C_mkl[ i ].y - (double)C_phigemm[ i ].y );
				if (tmp_error > MAX_ERROR ) {
					//				fprintf( stdout, "\nC_mkl[ %d ].y=%15.13e  C_phigemm[ %d ].y=%15.13e\n", i, C_mkl[ i ].y, i, C_phigemm[ i ].y);fflush(stdout);
					//				exit(EXIT_FAILURE);
					error += tmp_error;
				}
			}

			//		if (error > MAX_ERROR ) {
			//			fprintf( stdout, "\n\t\t ERRORS DETECTED IN COMPARING GEMM EXECUTION (%g) > %g", (double) error, (double) MAX_ERROR);
			//			fflush(stdout);
			//		}
#else
			XTYPE error = 0;
			XTYPE tmp_error;

#pragma omp parallel for reduction (+ : error)
			for( i = 0; i < m * n ; i++ ) {
				tmp_error = (XTYPE) abs( (XTYPE)(C_mkl[ i ] - C_phigemm[ i ]) );
				if (tmp_error > MAX_ERROR ) {
					//				fprintf( stdout, "\nC_mkl[ %d ]=%g  C_phigemm[ %d ]=%g", i, C_mkl[ i ], i, C_phigemm[ i ]);fflush(stdout);
					//				exit(EXIT_FAILURE);
					error += tmp_error;
				}
			}

			//		if (error > MAX_ERROR ) {
			//			fprintf( stdout, "\n\t\t ERRORS DETECTED IN COMPARING GEMM EXECUTION (%g) > %g\n", (double) error, (double) MAX_ERROR);
			//			fflush(stdout);
			//		}
#endif
#else
			// Fake declaration for output purposes....
			double error = -1;
#endif
			//	   fprintf( stdout, "\n\n");
			//	   fflush(stdout);

			fprintf( stdout, "[%c%c]  phiGEMM ( %d CPU / %d GPUs ) phiGEMM: Elapsed time = %10.6f s - RPeak = %10.4f GFlop/s\t(Split = %.3f)\t errors: %c\n", transa[ count ], transb[ count ], atoi( getenv( "MKL_NUM_THREADS" ) ), nGPU, hybrid_time, ( 1.e-6 ) * PHIGEMM_FLOPS(( double ) m, ( double ) n, ( double ) k) / (hybrid_time*1000), currentSplitFactor, (error > 0 ? 'Y' : (error == 0 ? 'N' : 'X')) );
			fflush( stdout );
//			fprintf( stdout, "[%c%c] MKL (%2d) RPeak = %10.4f\t\tCUBLAS RPeak = %10.4f (kernel RPeak = %g)\t\tphiGEMM (GPU: %d, split: %.3f) RPeak = %10.4f [errors %c]\n\n",
//					transa[ count ], transb[ count ], atoi( getenv( "MKL_NUM_THREADS" ) ),
//					( 1.e-6 ) * PHIGEMM_FLOPS(( double ) m, ( double ) n, ( double ) k) / (cpu_time*1000),
//					( 1.e-6 ) * PHIGEMM_FLOPS(( double ) m, ( double ) n, ( double ) k) / (gpu_time*1000),
//					( 1.e-6 ) * PHIGEMM_FLOPS(( double ) m, ( double ) n, ( double ) k) / (kernel_time*1000),
//					nGPU, currentSplitFactor,
//					( 1.e-6 ) * PHIGEMM_FLOPS(( double ) m, ( double ) n, ( double ) k) / (hybrid_time*1000),
//					(error > 0 ? 'Y' : (error == 0 ? 'N' : 'X')) );
//			fflush( stdout );

		} while((currentSplitFactor += stepSplitFactor) <= upperSplitFactor);
		/* ----------------------------------------------------------- */

		/* end */
	}

#if defined __PERFORM_PHIGEMM_INIT
	for( i = 0 ; i < nGPU ; i++) {

		if( cudaSetDevice( devicesToBond[i] ) != cudaSuccess )
		{
			fprintf( stderr, "*** ERROR cudaSetDevice\n");
			exit( EXIT_FAILURE );
		}

		if( cudaFree( test_scratch[ i ] ) != cudaSuccess )
		{
			fprintf( stderr, "[device:%d] Error cudaFree.\n", i);
			exit( EXIT_FAILURE );
		}
	}
#endif

	/* RELEASE RESOURCES */
#if defined __PERFORM_PHIGEMM_INIT
	phiGemmShutdown();
#endif

	free( C_mkl );

#ifdef __PHITEST_MEM_PINNED
	cudaFreeHost( GPU_buffer_memory_ptr );
	if ( !(mem_gpu > memsize[ 0 ]) ) cudaFreeHost( C_cuda );
#else

#if defined __PHITEST_FORCE_PINNED

        /* the first call makes no sense */
//        tmp_error = (int) cuMemHostGetFlags(&tmp_flags, GPU_buffer_memory_ptr);
//        printf("[cuMemHostGetFlags] tmp_error=%d, tmp_flags=%d\n",tmp_error, tmp_flags); fflush(stdout);

        tmp_error = (int) cudaHostUnregister(GPU_buffer_memory_ptr);
        printf("[cuMemHostUnregister] tmp_error=%d\n", tmp_error); fflush(stdout);

        tmp_error = (int) cudaHostGetFlags(&tmp_flags, GPU_buffer_memory_ptr);
        printf("[cuMemHostGetFlags] tmp_error=%d, tmp_flags=%d\n",tmp_error, tmp_flags); fflush(stdout);
#endif

	free( GPU_buffer_memory_ptr );

	if ( !(mem_gpu > memsize[ 0 ]) ) free( C_cuda );
#endif


	return 0;
}
