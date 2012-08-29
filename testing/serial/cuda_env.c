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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cuda_env.h"
#include "phigemm.h"

phiTestCudaMemDevPtr dev_scratch_test;
phiTestCudaMemSizes cuda_memory_allocated;
phiTestCudaDevicesBond gpu_bonded;

int ngpus_per_process;

void initcudaenv_()
{
	int ierr = 0;
	int lNumDevicesThisNode = 0;
	int i;
	size_t free, total;

	cudaGetDeviceCount(&lNumDevicesThisNode);

	if (lNumDevicesThisNode == 0)
	{
		fprintf( stderr,"***ERROR*** no CUDA-capable devices were found on the machine.\n");
		exit(EXIT_FAILURE);
	}

	/* multi-GPU in serial calculations is allowed ONLY if CUDA >= 4.0 */
#if defined __PHIGEMM_MULTI_GPU
	ngpus_per_process = lNumDevicesThisNode;
#else
	ngpus_per_process = 1;
#endif

	for (i = 0; i < ngpus_per_process; i++) {
		gpu_bonded[i] = i;
	}

	for (i = 0; i < ngpus_per_process; i++) {

		/* query the real free memory, taking into account the "stack" */
		if ( cudaSetDevice(gpu_bonded[i]) != cudaSuccess) {
			printf("*** ERROR *** cudaSetDevice(%d) failed!", gpu_bonded[i] );
			exit(EXIT_FAILURE);
		}

		cuda_memory_allocated[i] = (size_t) 0;

		ierr = cudaMalloc ( (void**) &(dev_scratch_test[i]), cuda_memory_allocated[i] );
		if ( ierr != cudaSuccess) {
			fprintf( stderr, "\nError in (first zero) memory allocation [GPU %d] , program will be terminated!!! Bye...\n\n", gpu_bonded[i]);
			exit(EXIT_FAILURE);
		}

		cudaMemGetInfo(&free, &total);

#if defined __PHIGEMM_DEBUG
		printf("[GPU %d] before: %lu (total: %lu)\n", gpu_bonded[i], (unsigned long)free, (unsigned long)total); fflush(stdout);
#endif

		cuda_memory_allocated[i] = (size_t) (free * __SCALING_MEM_FACTOR__);

		ierr = cudaMalloc ( (void**) &(dev_scratch_test[i]), (size_t) cuda_memory_allocated[i] );
		if ( ierr != cudaSuccess) {
			fprintf( stderr, "\nError in memory allocation [GPU %d] , program will be terminated (%d)!!! Bye...\n\n", gpu_bonded[i], ierr );
			exit(EXIT_FAILURE);
		}

#if defined __PHIGEMM_DEBUG
		cudaMemGetInfo(&free, &total);
		printf("[GPU %d] after: %lu (total: %lu)\n", gpu_bonded[i], (unsigned long)free, (unsigned long)total); fflush(stdout);
#endif
	}

	// tag = 0 (fake value)
	phiGemmInit(ngpus_per_process , (phiTestCudaMemDevPtr*)&dev_scratch_test, (phiTestCudaMemSizes*)&cuda_memory_allocated, (int *)gpu_bonded, 0);
}


void closecudaenv_()
{
	int ierr = 0;
	int i;

	for (i = 0; i < ngpus_per_process; i++) {

		/* query the real free memory, taking into account the "stack" */
		if ( cudaSetDevice(gpu_bonded[i]) != cudaSuccess) {
			printf("*** ERROR *** cudaSetDevice(%d) failed!", gpu_bonded[i] );
			exit(EXIT_FAILURE);
		}

		ierr = cudaFree ( dev_scratch_test[i] );

		if(ierr != cudaSuccess) {
			fprintf( stderr, "\nError in memory release, program will be terminated!!! Bye...\n\n" );
			exit(EXIT_FAILURE);
		}
	}

	phiGemmShutdown();

}
