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

#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>

#include <time.h>
#include <sys/types.h>
#include <sys/time.h>

#include "phigemm.h"
#include "phigemm_auxiliary.h"

#if defined(__PHIGEMM_PROFILE)
const char base[] = "phigemm.profile";
#endif

#ifdef __cplusplus
extern "C"
{
#endif

static int is_phigemm_init = 0;
static int is_external_memory_alloc = 0;
static int is_internal_memory_alloc = 0;
static int is_internal_memory_probed = 0;

/* auxiliary */
int stringCmp( const void *a, const void *b)
{
	return strcmp((const char*)a,(const char*)b);
}

/* This routine computes the memory required to store the considered matrices */
size_t memOccupancy(int is_splitA, float split, int m_in, int n_in, int k_in) {

#if !defined(__PHIGEMM_GPUONLY)
	int m_split, n_split, tmp;

	if (is_splitA) {
		tmp = (m_in) * split;
		//		if (m_in < 128)
		m_split = tmp;
		//		else
		//			m_split = floor(tmp/64.0)*64;

		return ( m_split*k_in/phiGemmNumDevices + k_in*n_in + m_split*n_in/phiGemmNumDevices );

	} else {
		tmp = (n_in) * split;
		//		if (n_in < 128)
		n_split = tmp;
		//		else
		//			n_split = floor(tmp/64.0)*64;

		return( m_in*k_in + k_in*n_split/phiGemmNumDevices + m_in*n_split/phiGemmNumDevices );
	}
#else
	return ( m_in*k_in + k_in*n_in + m_in*n_in );
#endif
}

/* This routine computes the recursive split */
void bestFit(int is_splitA, float split, int m, int n, int k, int type_size, int *p1, int *p2) {

	size_t memsize_gpu = scratch_size[0] * phiGemmNumDevices;
	size_t mem_gpu = memOccupancy(is_splitA, split, m, n, k) * type_size;

	int tmp_m = m;
	int tmp_n = n;

	// This is ok, much better if we consider  padded matrices...
	const int step = 64;

#if 0
	/* repeat until the "new" matrices fit the GPU memory */
	while (mem_gpu > memsize_gpu) {
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

/* This routine returns the selected strategy for CPU-GPU splitting */
int cpuGPUheuristic(int m, int n, int k, char type) {

	double ratio_km = (double) k/m;
	double ratio_kn = (double) k/n;
	double threshold = SPLITK_FACTOR*2; // default 20

	double LOWER_LIMIT_NM = 64;
	double UPPER_LIMIT_NM = 256;
	double UPPER_LIMIT_K = 1025; // 1024 is a good dimension....

	/* 0: CPU-only
	 * 1: special-K
	 * 2: standard (split A or B)
	 */

	// Un-comment ONLY for debug/testing purposes...
	// return 2;

#if defined(__PHIGEMM_ENABLE_SPECIALK)
	//if (type == 'd' || type == 'z') {
	if (type == 'd') {

#if defined(__PHIGEMM_DEBUG_4)
		printf("[PHIGEMM_DEBUG][4] ratio_km=%f, ratio_kn=%f, threshold=%f\n", ratio_km, ratio_kn, threshold); fflush(stdout);
#endif

		// Matrices are small but not so small...
		if ( (n >= LOWER_LIMIT_NM) && (m >= LOWER_LIMIT_NM) ){
			// over the UPPER limit, they have to be rectangular...
			if ( ((n >= UPPER_LIMIT_K) && (m >= UPPER_LIMIT_K)) && ((ratio_km >= SPLITK_FACTOR) || (ratio_kn >= SPLITK_FACTOR)) )
				return 1;
			// below the UPPER limit, they have to be very rectangular...
			if ( ((n < UPPER_LIMIT_K) && (m < UPPER_LIMIT_K)) && ((ratio_km >= threshold) || (ratio_kn >= threshold)) )
				return 1;
		}
	}
#endif

	if ( (n < UPPER_LIMIT_NM) ||  (m < UPPER_LIMIT_NM) ) return 0;

	return 2;
}

// ----


/*
 * Name			: phiGemmIsInit
 * Description	: return if phiGEMM is initialized or not
 * Visibility	: public
 */
int phiGemmIsInit()
{
	return is_phigemm_init;
}


/*
 * Name			: phiGemmIsInternalMemAlloc
 * Description	: return if memory has been allocated internally by phiGEMM
 * Visibility	: phiGEMM only
 */
int phiGemmIsInternalMemAlloc()
{
	return is_internal_memory_alloc;
}


/*
 * Name			: phiGemmIsExternalMemAlloc
 * Description	: return if memory has been allocated externally by the caller
 * Visibility	: phiGEMM only
 */
int phiGemmIsExternalMemAlloc()
{
	return is_external_memory_alloc;
}


/*
 * Name			: phigemm_cclock
 * Description	: return time in milliseconds
 * Visibility	: phiGEMM only
 */
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


/*
 * Name			: phigemmSetSplitFactor
 * Description	: the method set the current value of a specified
 * 				  split factor {S, C, D, Z}
 * Visibility	: public
 */
void phigemmSetSplitFactor(float *x) {
#if defined(__PHIGEMM_EXPLICIT_SPLITFACTOR)
	float tmp,tmp2;
	int i;

	for ( i = 0 ; i < 4 ; i++ ) {
		/* 0:SGEMM, 1:DGEMM, 2: CGEMM, 3:ZGEMM */
		tmp =  (100.0f * x[i])/( 1.0f - x[i]);
		tmp2 = 100.0f;
		phiGemmSplitFactor[i] = tmp / (tmp + tmp2);
	}
#endif
	return;
}


/*
 * Name			: phigemmGetSplitFactor
 * Description	: the method returns the current value of a specified
 * 				  split factor {S, C, D, Z}
 * Visibility	: public
 */
float phigemmGetSplitFactor(int selection) {
#if defined(__PHIGEMM_EXPLICIT_SPLITFACTOR)
	return phiGemmSplitFactor[selection];
#else
	return phiGemmPrevSplitFactor[selection];
#endif
}


/*
 * Name			: phiGemmInitMemory
 * Description	: the method performs the phiGEMM memory allocation and initialization
 * 				: based on the parameters
 * Visibility	: this file only
 *
 */
void phiGemmInitMemory( phiGemmMemSizes* dev_memsize )
{
	unsigned int i;
	cudaError_t ierr;
	size_t total, free;

	// I do not even know how many memory is available on the device...

	for (i = 0; i < phiGemmNumDevices * NSTREAMS; i++) {

		if (scratch_size[ i ] == 0)
		{
			if(dev_memsize == NULL) {

				// Detect how much memory is available
				// Assuming a process has exclusive access to the GPU

				is_internal_memory_probed = 1;

				/* query the real free memory, taking into account the "stack" */
				if ( cudaSetDevice( deviceIds[i % phiGemmNumDevices]) != cudaSuccess) {
					printf("*** ERROR *** cudaSetDevice(%d) failed!", deviceIds[i % phiGemmNumDevices] ); fflush(stdout);
					exit(EXIT_FAILURE);
				}

				/* Perform the allocation */
				ierr = cudaMalloc ( (void**) &(dev_scratch[i]), (size_t) 0 );
				if ( ierr != cudaSuccess) {
					fprintf( stderr, "\nError in (first zero) memory allocation , program will be terminated!!! Bye...\n\n");
					exit(EXIT_FAILURE);
				}

				cudaMemGetInfo((size_t*)&free, (size_t*)&total);

				scratch_size[i] = (size_t) (((free * __SCALING_MEM_FACTOR__ ) * 16.0) / 16.0);

			} else {

					scratch_size[ i ] = ( *dev_memsize )[ i ];
			}
		}
	}

	// Allocate & Initialize

	for (i = 0; i < phiGemmNumDevices * NSTREAMS; i++) {
		/* query the real free memory, taking into account the "stack" */
		if ( cudaSetDevice( deviceIds[i % phiGemmNumDevices]) != cudaSuccess) {
			printf("*** ERROR *** cudaSetDevice(%d) failed!",  deviceIds[i % phiGemmNumDevices] ); fflush(stdout);
			exit(EXIT_FAILURE);
		}

		ierr = cudaMalloc ( (void**) &(dev_scratch[i % phiGemmNumDevices]), (size_t) scratch_size[ i ] );
		if ( ierr != cudaSuccess) {
			fprintf( stderr, "\nError in memory allocation, program will be terminated (%d)!!! Bye...\n\n", ierr );
			exit(EXIT_FAILURE);
		}

#if defined(__PHIGEMM_DEBUG)
		printf("\n\n[PHIGEMM_DEBUG] %lu Bytes of memory is allocated internally on GPU %d\n\n", (unsigned long)scratch_size[i], deviceIds[i]);
		fflush(stdout);
#endif

		/* Attempt to initialize CUBLAS */
		if ( cublasCreate( &phiHandles[ i ] ) != CUBLAS_STATUS_SUCCESS ) {
			printf("*** phiGEMM *** ERROR *** cublasInit() for device %d failed!\n",i);
			fflush(stdout);
			exit( EXIT_FAILURE );
		}

		if( cudaStreamCreate( &phiStreams[ i ] ) != CUBLAS_STATUS_SUCCESS ) {
			printf("*** phiGEMM *** ERROR *** creating stream %d for device %d failed!\n", i, i % phiGemmNumDevices);
			fflush(stdout);
			exit( EXIT_FAILURE );
		}
		cublasSetStream( phiHandles[ i ], phiStreams[ i ] );
	}

#if defined(__PHIGEMM_PROFILE)
	// printf("\n\n*** phiGEMM *** open the file \n\n");fflush(stdout);
	phiProfileFile = fopen (finalFileName, "a");
#endif

	is_internal_memory_alloc = 1;
	return;
}


/*
 * Name			: phiGemmInit
 * Description	: the method initialize the library, both GPU binding and
 * 				  memory allocation according to the parameters
 * 				  *** EXPECTED TO CALL ONLY ONCE ***
 * Visibility	: public
 */
void phiGemmInit( int nGPU, phiGemmMemDevPtr* dev_ptr, phiGemmMemSizes* dev_memsize, int * deviceToBond, int tag )
{
	unsigned int i;

#if defined(__PHIGEMM_PROFILE)
	char *value = NULL;
#endif

#if !defined(__PHIGEMM_CPUONLY)
	struct cudaDeviceProp deviceProp;
	int deviceCount;
#endif

#if defined(__PHIGEMM_PROFILE)
	/* Create file descriptor where store the profiling information */

	value = getenv("PHIGEMM_PROFILE_PREFIX");

	if (tag < 0) {
		if (value != NULL)
			sprintf(finalFileName, "%s.%s.csv", base, value);
		else
			sprintf(finalFileName, "%s.csv", base);
	} else {
		if (value != NULL)
			sprintf(finalFileName, "%s.%d.%s.csv", base, tag, value);
		else
			sprintf(finalFileName, "%s.%d.csv", base, tag);
	}
#endif

	/* Skip all the initialization: phiGEMM becomes a simple interface to CPU GEMM so it is possible
	 * to capture all the GEMM call and profile them */
#if !defined(__PHIGEMM_CPUONLY)

	if ( is_phigemm_init == 1 )
		return;

	cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0) {
		printf("*** phiGEMM *** ERROR *** no CUDA-capable devices were found on node.\n");
		fflush(stdout);
		exit(EXIT_FAILURE);
	}

	if (nGPU > deviceCount) {
		printf("*** phiGEMM *** ERROR *** Requested %d devices, found on the node only %d. Initialization fails!\n", nGPU, deviceCount);
		fflush(stdout);
		exit(EXIT_FAILURE);
	}

	phiGemmNumDevices = nGPU;

	is_internal_memory_probed = 0;

	/* Initialize internal phiGEMM data structures */
	for( i = 0; i < phiGemmNumDevices * NSTREAMS; i++ )
	{
		dev_scratch[ i ] = NULL;
		scratch_size[ i ] = 0;
		phiHandles[ i ] = NULL;
		phiStreams[ i ] = NULL;
	}

	/* Read environment PHI_* variables (this reading override the default */
	readEnv();

	/* Assign GPU devices to process(es) */
	for (i = 0; i < phiGemmNumDevices * NSTREAMS; i++) {
		deviceIds[i] = deviceToBond[i % phiGemmNumDevices];
	}

	/* No memory pointer is provided -> Initialize the memory */
	if(dev_memsize != NULL) {

		// is_internal_memory_probed = 0;

		for (i = 0; i < phiGemmNumDevices * NSTREAMS; i++) {

			scratch_size[ i ] = ( *dev_memsize )[ i % phiGemmNumDevices ] / NSTREAMS;
		}
	}

	/* No memory pointer is provided -> Initialize the memory */
	if(dev_ptr != NULL) {

		for (i = 0; i < phiGemmNumDevices * NSTREAMS; i++) {

			// scratch_size[ i ] = ( *dev_memsize )[ i % phiGemmNumDevices ] / NSTREAMS;

			/* SAFE pointer operation! Remember that void pointers cannot be
			 * directly dereferenced because 'void' is NOT a real type! */

			/// THIS OPERATION IS WEIRD ///
			size_t offset = ( i / phiGemmNumDevices ) * scratch_size[ i ];
			char * tmp_ptr = (char*) ( ( *dev_ptr )[ i % phiGemmNumDevices ] );
			dev_scratch[ i ] = (void*) (tmp_ptr + offset) ;

#if defined(__PHIGEMM_DEBUG)
			printf("[PHIGEMM_DEBUG] %lu Bytes of memory is allocated externally on GPU %d\n", (unsigned long)scratch_size[i], deviceIds[i]);
			fflush(stdout);
#endif
		}

		for (i = 0; i < phiGemmNumDevices * NSTREAMS; i++) {

			/* Attempt to establish a runtime API context */
			if ( cudaSetDevice( deviceIds[i % phiGemmNumDevices]) != cudaSuccess) {
				printf("*** phiGEMM *** ERROR *** cudaSetDevice(%d) failed!\n",i );
				fflush(stdout);
				exit( EXIT_FAILURE );
			}

			/* Attempt to initialize CUBLAS */
			if ( cublasCreate( &phiHandles[ i ] ) != CUBLAS_STATUS_SUCCESS ) {
				printf("*** phiGEMM *** ERROR *** cublasInit() for device %d failed!\n",i);
				fflush(stdout);
				exit( EXIT_FAILURE );
			}

			if( cudaStreamCreate( &phiStreams[ i ] ) != CUBLAS_STATUS_SUCCESS ) {
				printf("*** phiGEMM *** ERROR *** creating stream %d for device %d failed!\n", i, i % phiGemmNumDevices);
				fflush(stdout);
				exit( EXIT_FAILURE );
			}
			cublasSetStream( phiHandles[ i ], phiStreams[ i ] );
		}

#if defined(__PHIGEMM_PROFILE)
		// printf("\n\n*** phiGEMM *** open the file \n\n");fflush(stdout);
		phiProfileFile = fopen (finalFileName, "a");
#endif
		is_external_memory_alloc = 1;
	}

	/* set the initialization flag */
	is_phigemm_init = 1;

	return;

#else

#if defined(__PHIGEMM_PROFILE)
	printf("\n\n*** phiGEMM *** open the file \n\n");fflush(stdout);
	phiProfileFile = fopen (finalFileName, "a");
#endif

	return;

#endif
}


/*
 * Name			: phiGemmInitMemory
 * Description	: the method performs the memory allocation on the GPU card
 * Visibility	: public
 */
void phiGemmShutdown()
{
	int i;

	/* Skip all the initialization: phiGEMM becomes a simple interface to CPU GEMM so it is possible
	 * to capture all the GEMM call and profile them */
#if !defined(__PHIGEMM_CPUONLY)

#if defined(__PHIGEMM_DEBUG)
	printf("[PHIGEMM_DEBUG] *** shutdown *** is_phigemm_init:%d, is_external_memory_alloc:%d, is_internal_memory_alloc:%d, devices: %d\n",is_phigemm_init, is_external_memory_alloc, is_internal_memory_alloc, phiGemmNumDevices);
	fflush(stdout);
#endif

	if ( !is_phigemm_init )
		return;

	if ( phiGemmIsExternalMemAlloc() ){

		for (i = 0; i < phiGemmNumDevices ; i++) {

			/* Attempt to establish a runtime API context */
			if ( cudaSetDevice( deviceIds[i % phiGemmNumDevices] ) != cudaSuccess) {
				printf("*** phiGEMM: *** ERROR *** cudaSetDevice(%d) failed!\n",i);
				exit(EXIT_FAILURE);
			}

			cudaStreamDestroy( phiStreams[ i ] );
			cublasDestroy( phiHandles[ i ]);

			is_external_memory_alloc = 0;
			is_phigemm_init = 0;

#if defined(__PHIGEMM_PROFILE)
			// printf("\n\n*** phiGEMM *** close the file \n\n");fflush(stdout);
			fclose (phiProfileFile);
#endif

		}
	}

	if ( phiGemmIsInternalMemAlloc() ){

		for ( i = 0; i < phiGemmNumDevices; i++ ){

			/* Attempt to establish a runtime API context */
			if ( cudaSetDevice( deviceIds[i % phiGemmNumDevices] ) != cudaSuccess) {
				printf("*** phiGEMM: *** ERROR *** cudaSetDevice(%d) failed!\n",i);
				exit(EXIT_FAILURE);
			}

			if (  cudaFree(dev_scratch[i]) != cudaSuccess) {
				printf("*** phiGEMM: *** ERROR *** cudaFree(%d) failed!\n",i);
				// exit(EXIT_FAILURE);
			}

			cudaStreamDestroy( phiStreams[ i ] );
			cublasDestroy( phiHandles[ i ]);

			dev_scratch[ i ] = NULL;
			if (is_internal_memory_probed) {
				scratch_size[ i ] = 0;
			}
			phiHandles[ i ] = NULL;
			phiStreams[ i ] = NULL;
		}

		is_internal_memory_alloc = 0;
	}

	return;

#else

#if defined(__PHIGEMM_PROFILE)
	// printf("\n\n*** phiGEMM *** close the file \n\n");fflush(stdout);
	fclose (phiProfileFile);
#endif

	return;
#endif

}

void phiGemmSetAvaiableScratchSpace(int gpu_id, size_t new_dev_memsize) {
	scratch_size[ deviceIds[gpu_id] ] = (size_t) new_dev_memsize;

#if defined(__PHIGEMM_DEBUG)
	printf("[PHIGEMM_DEBUG] %lu Bytes of GPU memory available %d\n", (unsigned long)scratch_size[gpu_id], deviceIds[gpu_id]);
	fflush(stdout);
#endif
}

/* ------------ FORTRAN INTERFACES FOR PHIGEMM PUBLIC METHODS -------------- */
void phigemminit_(int nGPU, phiGemmMemDevPtr* ptr, phiGemmMemSizes* dev_memsize, int * deviceToBond, int tag ){ phiGemmInit( nGPU, ptr, dev_memsize, deviceToBond, tag); }

void phigemmshutdown_(){ phiGemmShutdown(); }

int phigemmisinit_(){return phiGemmIsInit();}

void phigemmsetsplitfactor_(float *x) { phigemmSetSplitFactor(x); }

void phiremmsetavaiablescratchspace_(int gpu_id, size_t new_dev_memsize) { phiGemmSetAvaiableScratchSpace(gpu_id, new_dev_memsize); }

/* ------------------------------------------------------------------------- */

#ifdef __cplusplus
}
#endif
