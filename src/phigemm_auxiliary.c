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

#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>

#include <time.h>
#include <sys/types.h>
#include <sys/time.h>

#include "phigemm.h"
#include "phigemm_auxiliary.h"

#define THRESHOLD 2

#ifdef __cplusplus
extern "C"
{
#endif

static int is_phigemm_init = 0;
static int is_external_memory_alloc = 0;
static int is_internal_memory_alloc = 0;
static int is_internal_memory_probed = 0;

struct phiGemmHandler myPhiGemmHdl = {
		.SPLITK_FACTOR  = __SPLITK_FACTOR,
		.SPLITK_GEMM   = __SPLITK_GEMM,
		.LOWER_LIMIT    = __LOWER_LIMIT,
		.UPPER_LIMIT_NM = __UPPER_LIMIT_NM,
		.UPPER_LIMIT_K  = __UPPER_LIMIT_K
};


int stringCmp( const void *a, const void *b)
{
	return strcmp((const char*)a,(const char*)b);
}

size_t memOccupancy(int is_splitA, float split, int m_in, int n_in, int k_in) {

#if defined(__PHIGEMM_GPUONLY)
	return ( m_in*k_in + k_in*n_in + m_in*n_in );
#else
	int m_split, n_split, tmp;

	if (is_splitA) {
		tmp = (m_in) * split;
		//		if (m_in < 128)
		m_split = tmp;
		//		else
		//			m_split = floor(tmp/64.0)*64;

		return ( m_split*k_in + k_in*n_in + m_split*n_in );

	} else {
		tmp = (n_in) * split;
		//		if (n_in < 128)
		n_split = tmp;
		//		else
		//			n_split = floor(tmp/64.0)*64;

		return( m_in*k_in + k_in*n_split + m_in*n_split );
	}
#endif
}

void bestFit(int is_splitA, float split, int m, int n, int k, int type_size, int *p1, int *p2) {

	size_t mem_gpu = memOccupancy(is_splitA, split, m, n, k) * type_size;

#if 0
	int tmp_m = m;
	int tmp_n = n;

	// This is ok, much better if we consider  padded matrices...
	const int step = 64;

	/* repeat until the "new" matrices fit the GPU memory */
	while (mem_gpu > myPhiGemmHdl.smem) {
		if (is_splitA) {
			/* I can assume (to check!!!) that tmp_m is never too small ... */
			tmp_m = tmp_m - step;

			*p1 = tmp_m;
			*p2 = m - (*p1);

			/* A:(p1*split x k), B:(k x n), C: (p1*split x n)
			 * ---> do they fit the GPU memory?
			 */
			mem_gpu = memOccupancy(is_splitA, split, *p1, n, k) * type_size;
		} else {
			/* I can assume (to check!!!) that tmp_n is never too small ... */
			tmp_n = tmp_n - step;

			*p1 = tmp_n;
			*p2 = n - (*p1);

			/* A:(m x k), B:(k x p1*split), C: (m x p1*split)
			 * ---> do they fit the GPU memory?
			 */
			mem_gpu = memOccupancy(is_splitA, split, m, *p1, k) * type_size;
		}
#if defined(__PHIGEMM_DEBUG_4)
		fprintf( stdout,"[PHIGEMM_DEBUG][4] SPLITTING > p1: %d\tp2: %d\tsize (byte):%lu\n", *p1, *p2, mem_gpu); fflush(stdout);
#endif
	}
#else
	/* ORIGINAL */
	if (is_splitA) {
		*p1 = m/2;
		*p2 = m - (*p1);
	} else {
		*p1 = n/2;
		*p2 = n - (*p1);
	}
#endif

	return;
}

int cpuGPUheuristic(int m, int n, int k, char type)
{

	/* 0  : CPU-only
	 * 1  : special-K
	 * 2  : standard (split A or B) --> DEFAULT IS GPU ONLY !
	 */

#if defined(__PHIGEMM_ENABLE_SPECIALK)

	float RATIO_KM = (float) k/m;
	float RATIO_KN = (float) k/n;

	// Matrices are small but not so small...
	if ( (n >= myPhiGemmHdl.LOWER_LIMIT) && (m >= myPhiGemmHdl.LOWER_LIMIT) ){
		// over the UPPER limit, they have to be rectangular...
		if ( ((n >= myPhiGemmHdl.UPPER_LIMIT_K) && (m >= myPhiGemmHdl.UPPER_LIMIT_K)) && ((RATIO_KM >= myPhiGemmHdl.SPLITK_FACTOR) || (RATIO_KN >= myPhiGemmHdl.SPLITK_FACTOR)) )
			return 1;
		// below the UPPER limit, they have to be very rectangular...
		if ( ((n < myPhiGemmHdl.UPPER_LIMIT_K) && (m < myPhiGemmHdl.UPPER_LIMIT_K)) && ((RATIO_KM >= THRESHOLD) || (RATIO_KN >= THRESHOLD)) )
			return 1;
	}
#endif

	if ( (n < myPhiGemmHdl.LOWER_LIMIT) ||  (m < myPhiGemmHdl.LOWER_LIMIT) || (k < myPhiGemmHdl.LOWER_LIMIT))
		return 0;

	return 2;
}
// ----

int phiGemmIsInternalMemAlloc()
{
 	return is_internal_memory_alloc;
}

int phiGemmIsExternalMemAlloc()
{
 	return is_external_memory_alloc;
}

int phiGemmIsInit()
{
	return is_phigemm_init;
}

double phigemm_cclock(void)
{
	struct timeval tv;
	struct timezone tz;
	double t;

	gettimeofday(&tv, &tz);

	t = (double)tv.tv_sec;
	t += ((double)tv.tv_usec)/1000000.0;

	return t;
}


void phigemmSetSplitFactor(float split_gemm) {

	// BE AWARE: A MANUAL CHANGE IS PERMANENT ACROSS THE ENTIRE EXECUTION

	float tmp_split_gemm;

	tmp_split_gemm =  (100.0f * split_gemm)/( 1.0f - split_gemm);

	myPhiGemmHdl.SPLIT  = tmp_split_gemm / (tmp_split_gemm + 100.0f);

	return;
}

void phiGemmInitMemory( size_t dev_memsize )
{
	unsigned int i;
	cudaError_t ierr;
	size_t total, free;

	if (myPhiGemmHdl.smem == 0)
	{
		if(dev_memsize == 0) {

			// Detect how much memory is available
			// Assuming a process has exclusive access to the GPU

			is_internal_memory_probed = 1;

			/* query the real free memory, taking into account the "stack" */
			if ( cudaSetDevice( myPhiGemmHdl.devId ) != cudaSuccess) {
				printf("*** ERROR *** cudaSetDevice(%d) failed!", myPhiGemmHdl.devId ); fflush(stdout);
				exit(EXIT_FAILURE);
			}

			/* Perform the allocation */
			ierr = cudaMalloc ( (void**) &myPhiGemmHdl.pmem, (size_t) 0 );
			if ( ierr != cudaSuccess) {
				fprintf( stderr, "\nError in (first zero) memory allocation , program will be terminated!!! Bye...\n\n");
				exit(EXIT_FAILURE);
			}

			cudaMemGetInfo((size_t*)&free, (size_t*)&total);

			myPhiGemmHdl.smem = (size_t) (((free * __SCALING_INIT_MEM ) * 16.0) / 16.0);

		} else {

			myPhiGemmHdl.smem  = dev_memsize ;
		}
	}

	// Allocate & Initialize

	/* query the real free memory, taking into account the "stack" */
	if ( cudaSetDevice( myPhiGemmHdl.devId ) != cudaSuccess) {
		printf("*** ERROR *** cudaSetDevice(%d) failed!",  myPhiGemmHdl.devId ); fflush(stdout);
		exit(EXIT_FAILURE);
	}

	ierr = cudaMalloc ( (void**) &myPhiGemmHdl.pmem, (size_t) myPhiGemmHdl.smem );
	if ( ierr != cudaSuccess) {
		fprintf( stderr, "\nError in memory allocation, program will be terminated (%d)!!! Bye...\n\n", ierr );
		exit(EXIT_FAILURE);
	}

#if defined(__PHIGEMM_DEBUG)
	printf("\n\n[PHIGEMM_DEBUG] %lu Bytes of memory is allocated internally on GPU %d\n\n", (unsigned long)myPhiGemmHdl.smem, myPhiGemmHdl.devId);
	fflush(stdout);
#endif

	/* Attempt to initialize CUBLAS */
	if ( cublasCreate( &(myPhiGemmHdl.handle) ) != CUBLAS_STATUS_SUCCESS ) {
		printf("*** phiGEMM *** ERROR *** cublasInit() for device %d failed!\n",i);
		fflush(stdout);
		exit( EXIT_FAILURE );
	}

	if( cudaStreamCreate( &(myPhiGemmHdl.stream) ) != CUBLAS_STATUS_SUCCESS ) {
		printf("*** phiGEMM *** ERROR *** creating stream %d for device %d failed!\n", i, myPhiGemmHdl.devId);
		fflush(stdout);
		exit( EXIT_FAILURE );
	}
	cublasSetStream( myPhiGemmHdl.handle, myPhiGemmHdl.stream );

#if defined(__PHIGEMM_PROFILE)
	// printf("\n\n*** phiGEMM *** open the file \n\n");fflush(stdout);
	myPhiGemmHdl.profileFile = fopen (myPhiGemmEnv.filename, "a");
#endif

	is_internal_memory_alloc = 1;
	return;
}

void phiGemmInit( void* dev_ptr, size_t dev_memsize, int deviceToBond, int tag )
{
	unsigned int i;

	struct cudaDeviceProp deviceProp;
	int deviceCount;

	/* Read environment PHI_* variables (this reading override the default */
	readEnv(tag);

	/* Skip all the initialization: phiGEMM becomes a simple interface to CPU GEMM so it is possible
	 * to capture all the GEMM call and profile them */

	if ( is_phigemm_init == 1 )
		return;

	cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0) {
		printf("*** phiGEMM *** ERROR *** no CUDA-capable devices were found on node.\n");
		fflush(stdout);
		exit(EXIT_FAILURE);
	}

	/* Read environment PHI_* variables (this reading override the default */
	readEnv(tag);

	/* Skip all the initialization: phiGEMM becomes a simple interface to CPU GEMM so it is possible
	 * to capture all the GEMM call and profile them */

	if ( is_phigemm_init == 1 )
		return;

	is_internal_memory_probed = 0;

	/* Initialize internal phiGEMM data structures */
	myPhiGemmHdl.pmem = NULL;
	myPhiGemmHdl.smem = dev_memsize;
	myPhiGemmHdl.handle = NULL;
	myPhiGemmHdl.stream = NULL;
	myPhiGemmHdl.devId = deviceToBond;

	/* Attempt to establish a runtime API context */
	if ( cudaSetDevice( myPhiGemmHdl.devId) != cudaSuccess) {
		printf("*** phiGEMM *** ERROR *** cudaSetDevice(%d) failed!\n",i );
		fflush(stdout);
		exit( EXIT_FAILURE );
	}

	/* No memory pointer is provided -> Initialize the memory */
	if(dev_ptr != NULL) {

//			size_t offset = ( i / myPhiGemmEnv.numDevices ) * myPhiGemmHdl.smem[ i ];
//			char * tmp_ptr = (char*) ( ( *dev_ptr )[ i % myPhiGemmEnv.numDevices ] );
//			myPhiGemmHdl.pmem[ i ] = (void*) (tmp_ptr + offset) ;

#if defined(__PHIGEMM_DEBUG)
		printf("[PHIGEMM_DEBUG] %lu Bytes of memory is allocated externally on GPU %d\n", (unsigned long) myPhiGemmHdl.smem, myPhiGemmHdl.devId);
		fflush(stdout);
#endif

		/* Attempt to initialize CUBLAS */
		if ( cublasCreate( &(myPhiGemmHdl.handle) ) != CUBLAS_STATUS_SUCCESS ) {
			printf("*** phiGEMM *** ERROR *** cublasInit() for device %d failed!\n",i);
			fflush(stdout);
			exit( EXIT_FAILURE );
		}

		if( cudaStreamCreate( &(myPhiGemmHdl.stream) ) != CUBLAS_STATUS_SUCCESS ) {
			printf("*** phiGEMM *** ERROR *** creating stream for device %d failed!\n", myPhiGemmHdl.devId);
			fflush(stdout);
			exit( EXIT_FAILURE );
		}
		cublasSetStream( myPhiGemmHdl.handle, myPhiGemmHdl.stream );

#if defined(__PHIGEMM_PROFILE)
		myPhiGemmHdl.profileFile = fopen (myPhiGemmEnv.filename, "a");
#endif
		is_external_memory_alloc = 1;
	}

	/* set the initialization flag */
	is_phigemm_init = 1;

	return;
}


void phiGemmShutdown()
{
	int i;

	/* Skip all the initialization: phiGEMM becomes a simple interface to CPU GEMM so it is possible
	 * to capture all the GEMM call and profile them */

#if defined(__PHIGEMM_DEBUG)
	printf("[PHIGEMM_DEBUG] *** shutdown *** is_phigemm_init:%d, is_external_memory_alloc:%d, is_internal_memory_alloc:%d, devices: %d\n",is_phigemm_init, is_external_memory_alloc, is_internal_memory_alloc, myPhiGemmEnv.numDevices);
	fflush(stdout);
#endif

	if ( !is_phigemm_init )
		return;

	if ( phiGemmIsExternalMemAlloc() ){

		/* Attempt to establish a runtime API context */
		if ( cudaSetDevice( myPhiGemmHdl.devId ) != cudaSuccess) {
			printf("*** phiGEMM: *** ERROR *** cudaSetDevice(%d) failed!\n",i);
			exit(EXIT_FAILURE);
		}

		cudaStreamDestroy( myPhiGemmHdl.stream );
		cublasDestroy( myPhiGemmHdl.handle);

		is_external_memory_alloc = 0;
		is_phigemm_init = 0;

#if defined(__PHIGEMM_PROFILE)
		fclose (myPhiGemmHdl.profileFile);
#endif

	}

	if ( phiGemmIsInternalMemAlloc() ){

		/* Attempt to establish a runtime API context */
		if ( cudaSetDevice( myPhiGemmHdl.devId ) != cudaSuccess) {
			printf("*** phiGEMM: *** ERROR *** cudaSetDevice(%d) failed!\n",i);
			exit(EXIT_FAILURE);
		}

		if (  cudaFree(myPhiGemmHdl.pmem) != cudaSuccess) {
			printf("*** phiGEMM: *** ERROR *** cudaFree(%d) failed!\n",i);
			// exit(EXIT_FAILURE);
		}

		cudaStreamDestroy( myPhiGemmHdl.stream );
		cublasDestroy( myPhiGemmHdl.handle);

		myPhiGemmHdl.pmem = NULL;
		if (is_internal_memory_probed) {
			myPhiGemmHdl.smem = 0;
		}
		myPhiGemmHdl.handle = NULL;
		myPhiGemmHdl.stream = NULL;

		is_internal_memory_alloc = 0;
	}

	return;
}

/* ------------ FORTRAN INTERFACES FOR PHIGEMM PUBLIC METHODS -------------- */
void phigemminit_(void* dev_ptr, size_t dev_memsize, int deviceToBond, int tag  ){ phiGemmInit( dev_ptr, dev_memsize, deviceToBond, tag); }

void phigemmshutdown_(){ phiGemmShutdown(); }

int phigemmisinit_(){return phiGemmIsInit();}

void phigemmsetsplitfactor_(float split_gemm) { phigemmSetSplitFactor(split_gemm); }

/* ------------------------------------------------------------------------- */

#ifdef __cplusplus
}
#endif
