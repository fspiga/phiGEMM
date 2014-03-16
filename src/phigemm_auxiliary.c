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

struct phiGemmEnv myPhiGemmEnv;

#if !defined(__PHIGEMM_CPUONLY)
struct phiGemmHandler myPhiGemmHdl;
#endif

// C99-compatible initialization
struct phiGemmTuning myPhiGemmTng = {
		.SPLITK_FACTOR  = __SPLITK_FACTOR,
		.THRESHOLD      = (int) __SPLITK_FACTOR*1.5,
		.SPLITK_DGEMM   = __SPLITK_DGEMM,
		.SPLITK_ZGEMM   = __SPLITK_ZGEMM,
		.LOWER_LIMIT    = __LOWER_LIMIT,
		.UPPER_LIMIT_NM = __UPPER_LIMIT_NM,
		.UPPER_LIMIT_K  = __UPPER_LIMIT_K
};


/* auxiliary */
int stringCmp( const void *a, const void *b)
{
	return strcmp((const char*)a,(const char*)b);
}

#if !defined(__PHIGEMM_CPUONLY)
/* This routine computes the memory required to store the considered matrices */
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

		return ( m_split*k_in/myPhiGemmEnv.numDevices + k_in*n_in + m_split*n_in/myPhiGemmEnv.numDevices );

	} else {
		tmp = (n_in) * split;
		//		if (n_in < 128)
		n_split = tmp;
		//		else
		//			n_split = floor(tmp/64.0)*64;

		return( m_in*k_in + k_in*n_split/myPhiGemmEnv.numDevices + m_in*n_split/myPhiGemmEnv.numDevices );
	}
#endif
}
#endif

#if !defined(__PHIGEMM_CPUONLY)
/* This routine computes the recursive split */
void bestFit(int is_splitA, float split, int m, int n, int k, int type_size, int *p1, int *p2) {

	size_t memsize_gpu = myPhiGemmHdl.smem[0] * myPhiGemmEnv.numDevices;
	size_t mem_gpu = memOccupancy(is_splitA, split, m, n, k) * type_size;

#if 0
	int tmp_m = m;
	int tmp_n = n;

	// This is ok, much better if we consider  padded matrices...
	const int step = 64;

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
#endif

#if !defined(__PHIGEMM_CPUONLY)
/* This routine returns the selected strategy for CPU-GPU splitting */
int cpuGPUheuristic(int m, int n, int k, char type)
{

	/* 0  : CPU-only
	 * 1  : special-K
	 * 2  : standard (split A or B)
	 */

#if defined(__PHIGEMM_ENABLE_SPECIALK)

	float RATIO_KM = (float) k/m;
	float RATIO_KN = (float) k/n;

	if (type == 'd' || type == 'z') {

#if defined(__PHIGEMM_DEBUG_4)
		printf("[PHIGEMM_DEBUG][4] ratio_km=%f, ratio_kn=%f, threshold=%f\n", RATIO_KM, RATIO_KN, myPhiGemmTng.THRESHOLD); fflush(stdout);
#endif

		// Matrices are small but not so small...
		if ( (n >= myPhiGemmTng.LOWER_LIMIT) && (m >= myPhiGemmTng.LOWER_LIMIT) ){
			// over the UPPER limit, they have to be rectangular...
			if ( ((n >= myPhiGemmTng.UPPER_LIMIT_K) && (m >= myPhiGemmTng.UPPER_LIMIT_K)) && ((RATIO_KM >= myPhiGemmTng.SPLITK_FACTOR) || (RATIO_KN >= myPhiGemmTng.SPLITK_FACTOR)) )
				return 1;
			// below the UPPER limit, they have to be very rectangular...
			if ( ((n < myPhiGemmTng.UPPER_LIMIT_K) && (m < myPhiGemmTng.UPPER_LIMIT_K)) && ((RATIO_KM >= myPhiGemmTng.THRESHOLD) || (RATIO_KN >= myPhiGemmTng.THRESHOLD)) )
				return 1;
		}
	}
#endif

	if ( (n < myPhiGemmTng.LOWER_LIMIT) ||  (m < myPhiGemmTng.LOWER_LIMIT) || (k < myPhiGemmTng.LOWER_LIMIT)) return 0;

	return 2;
}
#endif

// ----


#if !defined(__PHIGEMM_CPUONLY)
/*
 * Name			: phiGemmIsInit
 * Description	: return if phiGEMM is initialized or not
 * Visibility	: public
 */
int phiGemmIsInit()
{
	return is_phigemm_init;
}
#endif

#if !defined(__PHIGEMM_CPUONLY)
/*
 * Name			: phiGemmIsInternalMemAlloc
 * Description	: return if memory has been allocated internally by phiGEMM
 * Visibility	: phiGEMM only
 */
int phiGemmIsInternalMemAlloc()
{
	return is_internal_memory_alloc;
}
#endif

#if !defined(__PHIGEMM_CPUONLY)
/*
 * Name			: phiGemmIsExternalMemAlloc
 * Description	: return if memory has been allocated externally by the caller
 * Visibility	: phiGEMM only
 */
int phiGemmIsExternalMemAlloc()
{
	return is_external_memory_alloc;
}
#endif

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


#if !defined(__PHIGEMM_CPUONLY)
/*
 * Name			: phigemmSetSplitFactor
 * Description	: the method set the current value of a specified
 * 				  split factor {S, C, D, Z}
 * Visibility	: public
 */
void phigemmSetSplitFactor(float split_dgemm, float split_zgemm) {
#if defined(__PHIGEMM_EXPLICIT_SPLITFACTOR)
	/* 0:DGEMM, 1:ZGEMM */
	float tmp_split_dgemm, tmp_split_zgemm;

	tmp_split_dgemm =  (100.0f * split_dgemm)/( 1.0f - split_dgemm);
	tmp_split_zgemm =  (100.0f * split_zgemm)/( 1.0f - split_zgemm);

	myPhiGemmTng.split[0] = tmp_split_dgemm / (tmp_split_dgemm + 100.0f);
	myPhiGemmTng.split[1] = tmp_split_zgemm / (tmp_split_zgemm + 100.0f);


	}
#endif
	return;
}
#endif

#if !defined(__PHIGEMM_CPUONLY)
/*
 * Name			: phigemmGetSplitFactor
 * Description	: the method returns the current value of a specified
 * 				  split factor {D, Z}
 * Visibility	: public
 */
float phigemmGetSplitFactor(int selection) {
#if defined(__PHIGEMM_EXPLICIT_SPLITFACTOR)
	return myPhiGemmTng.split[selection];
#else
	return myPhiGemmTng.prevSplit[selection];
#endif
}
#endif

#if !defined(__PHIGEMM_CPUONLY)
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

	for (i = 0; i < myPhiGemmEnv.numDevices * NSTREAMS; i++) {

		if (myPhiGemmHdl.smem[ i ] == 0)
		{
			if(dev_memsize == NULL) {

				// Detect how much memory is available
				// Assuming a process has exclusive access to the GPU

				is_internal_memory_probed = 1;

				/* query the real free memory, taking into account the "stack" */
				if ( cudaSetDevice( myPhiGemmHdl.devId[i % myPhiGemmEnv.numDevices]) != cudaSuccess) {
					printf("*** ERROR *** cudaSetDevice(%d) failed!", myPhiGemmHdl.devId[i % myPhiGemmEnv.numDevices] ); fflush(stdout);
					exit(EXIT_FAILURE);
				}

				/* Perform the allocation */
				ierr = cudaMalloc ( (void**) &(myPhiGemmHdl.pmem[i]), (size_t) 0 );
				if ( ierr != cudaSuccess) {
					fprintf( stderr, "\nError in (first zero) memory allocation , program will be terminated!!! Bye...\n\n");
					exit(EXIT_FAILURE);
				}

				cudaMemGetInfo((size_t*)&free, (size_t*)&total);

				myPhiGemmHdl.smem[i] = (size_t) (((free * __SCALING_INIT_MEM ) * 16.0) / 16.0);

			} else {

				myPhiGemmHdl.smem[ i ] = ( *dev_memsize )[ i ];
			}
		}
	}

	// Allocate & Initialize

	for (i = 0; i < myPhiGemmEnv.numDevices * NSTREAMS; i++) {
		/* query the real free memory, taking into account the "stack" */
		if ( cudaSetDevice( myPhiGemmHdl.devId[i % myPhiGemmEnv.numDevices]) != cudaSuccess) {
			printf("*** ERROR *** cudaSetDevice(%d) failed!",  myPhiGemmHdl.devId[i % myPhiGemmEnv.numDevices] ); fflush(stdout);
			exit(EXIT_FAILURE);
		}

		ierr = cudaMalloc ( (void**) &(myPhiGemmHdl.pmem[i % myPhiGemmEnv.numDevices]), (size_t) myPhiGemmHdl.smem[ i ] );
		if ( ierr != cudaSuccess) {
			fprintf( stderr, "\nError in memory allocation, program will be terminated (%d)!!! Bye...\n\n", ierr );
			exit(EXIT_FAILURE);
		}

#if defined(__PHIGEMM_DEBUG)
		printf("\n\n[PHIGEMM_DEBUG] %lu Bytes of memory is allocated internally on GPU %d\n\n", (unsigned long)myPhiGemmHdl.smem[i], myPhiGemmHdl.devId[i]);
		fflush(stdout);
#endif

		/* Attempt to initialize CUBLAS */
		if ( cublasCreate( &(myPhiGemmHdl.handle[ i ]) ) != CUBLAS_STATUS_SUCCESS ) {
			printf("*** phiGEMM *** ERROR *** cublasInit() for device %d failed!\n",i);
			fflush(stdout);
			exit( EXIT_FAILURE );
		}

		if( cudaStreamCreate( &(myPhiGemmHdl.stream[ i ]) ) != CUBLAS_STATUS_SUCCESS ) {
			printf("*** phiGEMM *** ERROR *** creating stream %d for device %d failed!\n", i, i % myPhiGemmEnv.numDevices);
			fflush(stdout);
			exit( EXIT_FAILURE );
		}
		cublasSetStream( myPhiGemmHdl.handle[ i ], myPhiGemmHdl.stream[ i ] );
	}

#if defined(__PHIGEMM_PROFILE)
	// printf("\n\n*** phiGEMM *** open the file \n\n");fflush(stdout);
	myPhiGemmEnv.profileFile = fopen (myPhiGemmEnv.filename, "a");
#endif

	is_internal_memory_alloc = 1;
	return;
}
#endif

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
			sprintf(myPhiGemmEnv.filename, "%s.%s.csv", base, value);
		else
			sprintf(myPhiGemmEnv.filename, "%s.csv", base);
	} else {
		if (value != NULL)
			sprintf(myPhiGemmEnv.filename, "%s.%d.%s.csv", base, tag, value);
		else
			sprintf(myPhiGemmEnv.filename, "%s.%d.csv", base, tag);
	}
#endif

	/* Read environment PHI_* variables (this reading override the default */
	readEnv();

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

	myPhiGemmEnv.numDevices = nGPU;

	is_internal_memory_probed = 0;

	/* Initialize internal phiGEMM data structures */
	for( i = 0; i < myPhiGemmEnv.numDevices * NSTREAMS; i++ )
	{
		myPhiGemmHdl.pmem[ i ] = NULL;
		myPhiGemmHdl.smem[ i ] = 0;
		myPhiGemmHdl.handle[ i ] = NULL;
		myPhiGemmHdl.stream[ i ] = NULL;
	}

	/* Assign GPU devices to process(es) */
	for (i = 0; i < myPhiGemmEnv.numDevices * NSTREAMS; i++) {
		myPhiGemmHdl.devId[i] = deviceToBond[i % myPhiGemmEnv.numDevices];
	}

	/* No memory pointer is provided -> Initialize the memory */
	if(dev_memsize != NULL) {

		// is_internal_memory_probed = 0;

		for (i = 0; i < myPhiGemmEnv.numDevices * NSTREAMS; i++) {

			myPhiGemmHdl.smem[ i ] = ( *dev_memsize )[ i % myPhiGemmEnv.numDevices ] / NSTREAMS;
		}
	}

	/* No memory pointer is provided -> Initialize the memory */
	if(dev_ptr != NULL) {

		for (i = 0; i < myPhiGemmEnv.numDevices * NSTREAMS; i++) {

			// scratch_size[ i ] = ( *dev_memsize )[ i % phiGemmNumDevices ] / NSTREAMS;

			/* SAFE pointer operation! Remember that void pointers cannot be
			 * directly dereferenced because 'void' is NOT a real type! */

			/// THIS OPERATION IS WEIRD ///
			size_t offset = ( i / myPhiGemmEnv.numDevices ) * myPhiGemmHdl.smem[ i ];
			char * tmp_ptr = (char*) ( ( *dev_ptr )[ i % myPhiGemmEnv.numDevices ] );
			myPhiGemmHdl.pmem[ i ] = (void*) (tmp_ptr + offset) ;

#if defined(__PHIGEMM_DEBUG)
			printf("[PHIGEMM_DEBUG] %lu Bytes of memory is allocated externally on GPU %d\n", (unsigned long) myPhiGemmHdl.smem[i], myPhiGemmHdl.devId[i]);
			fflush(stdout);
#endif
		}

		for (i = 0; i < myPhiGemmEnv.numDevices * NSTREAMS; i++) {

			/* Attempt to establish a runtime API context */
			if ( cudaSetDevice( myPhiGemmHdl.devId[i % myPhiGemmEnv.numDevices]) != cudaSuccess) {
				printf("*** phiGEMM *** ERROR *** cudaSetDevice(%d) failed!\n",i );
				fflush(stdout);
				exit( EXIT_FAILURE );
			}

			/* Attempt to initialize CUBLAS */
			if ( cublasCreate( &(myPhiGemmHdl.handle[ i ]) ) != CUBLAS_STATUS_SUCCESS ) {
				printf("*** phiGEMM *** ERROR *** cublasInit() for device %d failed!\n",i);
				fflush(stdout);
				exit( EXIT_FAILURE );
			}

			if( cudaStreamCreate( &(myPhiGemmHdl.stream[ i ]) ) != CUBLAS_STATUS_SUCCESS ) {
				printf("*** phiGEMM *** ERROR *** creating stream %d for device %d failed!\n", i, i % myPhiGemmEnv.numDevices);
				fflush(stdout);
				exit( EXIT_FAILURE );
			}
			cublasSetStream( myPhiGemmHdl.handle[ i ], myPhiGemmHdl.stream[ i ] );
		}

#if defined(__PHIGEMM_PROFILE)
		// printf("\n\n*** phiGEMM *** open the file \n\n");fflush(stdout);
		myPhiGemmEnv.profileFile = fopen (myPhiGemmEnv.filename, "a");
#endif
		is_external_memory_alloc = 1;
	}

	/* set the initialization flag */
	is_phigemm_init = 1;

	return;

#else

#if defined(__PHIGEMM_PROFILE)
	//printf("\n\n*** phiGEMM *** open the file \n\n");fflush(stdout);
	myPhiGemmEnv.profileFile = fopen (myPhiGemmEnv.filename, "a");
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
	printf("[PHIGEMM_DEBUG] *** shutdown *** is_phigemm_init:%d, is_external_memory_alloc:%d, is_internal_memory_alloc:%d, devices: %d\n",is_phigemm_init, is_external_memory_alloc, is_internal_memory_alloc, myPhiGemmEnv.numDevices);
	fflush(stdout);
#endif

	if ( !is_phigemm_init )
		return;

	if ( phiGemmIsExternalMemAlloc() ){

		for (i = 0; i < myPhiGemmEnv.numDevices ; i++) {

			/* Attempt to establish a runtime API context */
			if ( cudaSetDevice( myPhiGemmHdl.devId[i % myPhiGemmEnv.numDevices] ) != cudaSuccess) {
				printf("*** phiGEMM: *** ERROR *** cudaSetDevice(%d) failed!\n",i);
				exit(EXIT_FAILURE);
			}

			cudaStreamDestroy( myPhiGemmHdl.stream[ i ] );
			cublasDestroy( myPhiGemmHdl.handle[ i ]);

			is_external_memory_alloc = 0;
			is_phigemm_init = 0;

#if defined(__PHIGEMM_PROFILE)
			// printf("\n\n*** phiGEMM *** close the file \n\n");fflush(stdout);
			fclose (myPhiGemmEnv.profileFile);
#endif

		}
	}

	if ( phiGemmIsInternalMemAlloc() ){

		for ( i = 0; i < myPhiGemmEnv.numDevices; i++ ){

			/* Attempt to establish a runtime API context */
			if ( cudaSetDevice( myPhiGemmHdl.devId[i % myPhiGemmEnv.numDevices] ) != cudaSuccess) {
				printf("*** phiGEMM: *** ERROR *** cudaSetDevice(%d) failed!\n",i);
				exit(EXIT_FAILURE);
			}

			if (  cudaFree(myPhiGemmHdl.pmem[i]) != cudaSuccess) {
				printf("*** phiGEMM: *** ERROR *** cudaFree(%d) failed!\n",i);
				// exit(EXIT_FAILURE);
			}

			cudaStreamDestroy( myPhiGemmHdl.stream[ i ] );
			cublasDestroy( myPhiGemmHdl.handle[ i ]);

			myPhiGemmHdl.pmem[ i ] = NULL;
			if (is_internal_memory_probed) {
				myPhiGemmHdl.smem[ i ] = 0;
			}
			myPhiGemmHdl.handle[ i ] = NULL;
			myPhiGemmHdl.stream[ i ] = NULL;
		}

		is_internal_memory_alloc = 0;
	}

	return;

#else

#if defined(__PHIGEMM_PROFILE)
	// printf("\n\n*** phiGEMM *** close the file \n\n");fflush(stdout);
	fclose (myPhiGemmEnv.profileFile);
#endif

	return;
#endif

}

#if !defined(__PHIGEMM_CPUONLY)
void phiGemmSetAvaiableScratchSpace(int gpu_id, size_t new_dev_memsize) {
	myPhiGemmHdl.smem[ myPhiGemmHdl.devId[gpu_id] ] = (size_t) new_dev_memsize;

#if defined(__PHIGEMM_DEBUG)
	printf("[PHIGEMM_DEBUG] %lu Bytes of GPU memory available %d\n", (unsigned long)myPhiGemmHdl.smem[gpu_id], myPhiGemmHdl.devId[gpu_id]);
	fflush(stdout);
#endif
}
#endif

/* ------------ FORTRAN INTERFACES FOR PHIGEMM PUBLIC METHODS -------------- */
void phigemminit_(int nGPU, phiGemmMemDevPtr* ptr, phiGemmMemSizes* dev_memsize, int * deviceToBond, int tag ){ phiGemmInit( nGPU, ptr, dev_memsize, deviceToBond, tag); }

void phigemmshutdown_(){ phiGemmShutdown(); }

#if !defined(__PHIGEMM_CPUONLY)
int phigemmisinit_(){return phiGemmIsInit();}

void phigemmsetsplitfactor_(float split_dgemm, float split_zgemm) { phigemmSetSplitFactor(split_dgemm, split_zgemm); }

void phiremmsetavaiablescratchspace_(int gpu_id, size_t new_dev_memsize) { phiGemmSetAvaiableScratchSpace(gpu_id, new_dev_memsize); }
#endif
/* ------------------------------------------------------------------------- */

#ifdef __cplusplus
}
#endif
