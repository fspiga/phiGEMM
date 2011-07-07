/*
 * Copyright (C) 2010-2011 Irish Centre for High-End Computing (ICHEC)
 *
 * This file is distributed under the terms of the
 * GNU General Public License. See the file `License'
 * in the root directory of the present distribution,
 * or http://www.gnu.org/copyleft/gpl.txt .
 *
 * author(s):	Philip Yang   (phi@cs.umd.edu)
 * 				Filippo Spiga (filippo.spiga@ichec.ie)
 * 				Ivan Girotto  (ivan.girotto@ichec.ie)
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>

#ifdef __PHIGEMM_PARA
#include <mpi.h>
#endif

#include <time.h>

#include <sys/types.h>
#include <sys/time.h>

#include "phigemm.h"
#include "phigemm_auxiliary.h"

#include "cuda.h"

#include "cublas_api.h"
#include <cuda_runtime.h>
#include "cublas_v2.h"


#ifdef __cplusplus
extern "C"
{
#endif

static int is_phigemm_init = 0;
static int myrank = 0;

cudaStream_t  phiStreams[ NSTREAM_PER_DEVICE * MAX_GPUS ];
cublasHandle_t phiHandles[ NSTREAM_PER_DEVICE * MAX_GPUS ];
int phiGemmNumDevices;

float phiGemmSplitFactor[3];
phiGemmMemDevPtr dev_scratch;
phiGemmMemSizes scratch_size;
phiGemmDeviceIds deviceIds;

static int is_alloc_external = 0;

#ifdef __PHIGEMM_PROFILE
FILE *phiProfileFile;
#endif

int phiGemmIsInit()
{
	return is_phigemm_init;
}

int phiGemmGetRank()
{
	return myrank;
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


TimeStruct get_current_time(void)
{
	static struct timeval  time_val;
	static struct timezone time_zone;

	TimeStruct time;

	gettimeofday(&time_val, &time_zone);

	time.sec  = time_val.tv_sec;
	time.usec = time_val.tv_usec;
	return (time);
}


double GetTimerValue(TimeStruct time_1, TimeStruct time_2)
{
	int sec, usec;

	sec  = time_2.sec  - time_1.sec;
	usec = time_2.usec - time_1.usec;

	return (1000.*(double)(sec) + (double)(usec) * 0.001);
}


int stringCmp( const void *a, const void *b)
{
	return strcmp((const char*)a,(const char*)b);
}


void phigemmSetSplitFactor(float *x) {
#ifdef __PHIGEMM_EXPLICIT_SPLITFACTOR
	float tmp,tmp2;
	int i;

	for ( i = 0 ; i < 3 ; i++ ) {
		/* 0:SGEMM, 1:DGEMM, 2:ZGEMM */
		tmp =  (100.0f * x[i])/( 1.0f - x[i]);
		tmp2 = 100.0f;
		phiGemmSplitFactor[i] = tmp / (tmp + tmp2);
	}
#else
	/* read from environment */
	estmSplitFactor("xxx", 'n', 'n');
#endif
	return;
}


void estmSplitFactor(const char* optype, char transa, char transb)
{
	float envar_split;
	char *value = NULL;

	/* split factor may vary between S/D/Z GEMMs */

	/* SGEMM */
	value = getenv("PHI_SGEMM_SPLIT");
	if (value != NULL)
	{
		envar_split = atof(value);
#ifdef __PHIGEMM_DEBUG
		printf ("*** phiGEMM *** SGEMM split factor from environment variable: %f \n", envar_split);
#endif
	} else {
		/* Default split if no env variables are specified */
		envar_split = 0.85;
#ifdef __PHIGEMM_DEBUG
		printf ("*** phiGEMM *** SGEMM default split factor: %f \n", envar_split);
#endif
	}
	phiGemmSplitFactor[0] = envar_split;

	/* SGEMM */
	value = getenv("PHI_DGEMM_SPLIT");
	if (value != NULL)
	{
		envar_split = atof(value);
#ifdef __PHIGEMM_DEBUG
		printf ("*** phiGEMM *** DGEMM split factor from environment variable: %f \n", envar_split);
#endif
	} else {
		/* Default split if no env variables are specified */
		envar_split = 0.875;
#ifdef __PHIGEMM_DEBUG
		printf ("*** phiGEMM *** DGEMM default split factor: %f \n", envar_split);
#endif
	}
	phiGemmSplitFactor[1] = envar_split;

	/* ZGEMM */
	value = getenv("PHI_ZGEMM_SPLIT");
	if (value != NULL)
	{
		envar_split = atof(value);
#ifdef __PHIGEMM_DEBUG
		printf ("*** phiGEMM *** ZGEMM split factor from environment variable: %f \n", envar_split);
#endif
	} else {

		/* Default split if no env variables are specified */
		envar_split = 0.9;
#ifdef __PHIGEMM_DEBUG
		printf ("*** phiGEMM *** ZGEMM  default split factor: %f \n", envar_split);
#endif
	}
	phiGemmSplitFactor[2] = envar_split;

}


void phiGemmInit( int nGPU, phiGemmMemDevPtr* dev_ptr, phiGemmMemSizes* dev_memsize, int * deviceToBond )
{

	struct cudaDeviceProp deviceProp;
	unsigned int i;
	int deviceCount;

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

	if ( is_phigemm_init == 1 )
		return;

	is_alloc_external = 1;

#ifdef __PHIGEMM_DEBUG
	printf("*** phiGEMM *** %d GPUs detected.\n", phiGemmNumDevices);
	fflush(stdout);
#endif

	/* find the split factor */
#ifdef __PHIGEMM_EXPLICIT_SPLITFACTOR

#ifdef __PHIGEMM_DEBUG
	printf("*** phiGEMM *** The (explicit) split factors are: %g %g %g\n", phiGemmSplitFactor[0], phiGemmSplitFactor[1], phiGemmSplitFactor[2]);
	fflush(stdout);
#endif

#else

	/* Now there is only one generic split factor. Parameters are temporary ignored... */
	estmSplitFactor("xxx", 'n', 'n');

#ifdef __PHIGEMM_DEBUG
	printf("*** phiGEMM *** The (initial) split factor is %g\n", phiGemmSplitFactor);
	fflush(stdout);
#endif

#endif

	/* Init GPU data structures for managing multiGPU */
	for( i = 0; i < phiGemmNumDevices * NSTREAM_PER_DEVICE; i++ )
	{
		dev_scratch[ i ] = NULL;
		scratch_size[ i ] = 0;
		phiHandles[ i ] = NULL;
		phiStreams[ i ] = NULL;
	}

	for (i = 0; i < phiGemmNumDevices * NSTREAM_PER_DEVICE; i++) {

		/* Assign devices to processes
		 * note: one process may have assigned more than one device */
		deviceIds[i] = deviceToBond[i % phiGemmNumDevices];

		scratch_size[ i ] = ( *dev_memsize )[ i % phiGemmNumDevices ] / NSTREAM_PER_DEVICE;

		dev_scratch[ i ] = ( *dev_ptr )[ i % phiGemmNumDevices ] + ( ( i / phiGemmNumDevices ) * scratch_size[ i ] );


#ifdef __PHIGEMM_DEBUG
		printf("*** phiGEMM *** %lu Bytes of memory is allocated externally on GPU %d\n", (unsigned long)scratch_size[i], deviceIds[i]);
		fflush(stdout);
#endif
	}

	cudaError_t err;

	for (i = 0; i < phiGemmNumDevices * NSTREAM_PER_DEVICE; i++) {

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

	// NOT YET READY FOR MULTI-GPU
#if 0
	/* allocate scratch space for library on device */
	int err1;

	if ( !is_alloc_external )
	{
		if (*dev_memsize > 0) {

			scratch_size[i] = dev_memsize[0];

#ifdef __PHIGEMM_DEBUG
			printf("*** phiGEMM: %lu (%f G) Byte of the global memory requested to be allocated\n", (unsigned long)scratch_size, (double)scratch_size / OnEGiG);
			fflush(stdout);
#endif


		} else {

			size_t memsize_free, memsize_total;
			cudaMemGetInfo(&memsize_free, &memsize_total);

			if ( memsize_free < OnEGiG )
				scratch_size = (size_t) ceil(0.88 * (double) memsize_free);
			else
				scratch_size = (size_t) ceil(0.618 * (double) memsize_free);

#ifdef __PHIGEMM_DEBUG
			printf("*** phiGEMM: %lu (%f G) Byte of the global memory will be allocated\n", (unsigned long)scratch_size, (double)scratch_size / OnEGiG);
			fflush(stdout);
#endif
		}

		err1 = cudaMalloc((void*)dev_scratch[0], scratch_size);

		if ( err1 != 0 )
		{
			printf("*** phiGEMM: device memory allocation error: %d\n", err1);
			fflush(stdout);
#ifdef __PHIGEMM_PARA
			MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
#else
			exit(EXIT_FAILURE);
#endif
		}

	} else {
		/* using external device pointer
		 * assume the memory is already allocated by the user */
#ifdef __PHIGEMM_DEBUG
		printf ("*** phiGEMM: using external device pointer\n");
		fflush(stdout);
#endif
	}
#endif

	/* set the initialization flag */
	is_phigemm_init = 1;

#ifdef __PHIGEMM_PROFILE
	phiProfileFile = fopen ("phigemm.profile", "a");
#endif
}

void phiGemmShutdown()
{
	int i;

	if ( !is_phigemm_init )
		return;

	for (i = 0; i < phiGemmNumDevices * NSTREAM_PER_DEVICE; i++) {

		/* Attempt to establish a runtime API context */
		if ( cudaSetDevice( i % phiGemmNumDevices) != cudaSuccess) {
			printf("*** phiGEMM: *** ERROR *** cudaSetDevice(%d) failed!\n",i);
			exit(EXIT_FAILURE);
		}

		cudaStreamDestroy( phiStreams[ i ] );
		cublasDestroy( phiHandles[ i ]);
	}

	for ( i = 0; i < phiGemmNumDevices; i++ ){

		if ( !is_alloc_external )
			cudaFree(dev_scratch[i]);
	}

	is_phigemm_init = 0;

#ifdef __PHIGEMM_PROFILE
	fclose (phiProfileFile);
#endif
}


void phigemminit_(int nGPU, phiGemmMemDevPtr* ptr, phiGemmMemSizes* dev_memsize, int * deviceToBond ){ phiGemmInit( nGPU, ptr, dev_memsize, deviceToBond); }

void phigemmshutdown_(){ phiGemmShutdown(); }

int phigemmisinit_(){return phiGemmIsInit();}
int phigemmgetrank_(){return phiGemmGetRank();}


#ifdef __cplusplus
}
#endif
