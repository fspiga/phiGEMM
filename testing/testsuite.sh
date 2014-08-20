#!/bin/bash

NUMA_CTL="numactl -m 1 -c 1"

export OMP_NUM_THREADS=6 
export MKL_NUM_THREADS=6

sleep 1

export CUDA_VISIBLE_DEVICES=0

echo "'\nTesting DGEMM\n"
make TEST_DATATYPE_FLAGS=-D__CUDA_TYPE_DOUBLE
env LD_LIBRARY_PATH=../lib:${LD_LIBRARY_PATH} ${NUMA_CTL} ../bin/single_test-dgemm.x 1 N N 4096 4096 4096 0.95 0.995 0.01

sleep 1

echo "'\nTesting ZGEMM\n"
make TEST_DATATYPE_FLAGS=-D__CUDA_TYPE_DOUBLE_COMPLEX
env LD_LIBRARY_PATH=../lib:${LD_LIBRARY_PATH} ${NUMA_CTL} ../bin/single_test-zgemm.x 1 N N 4096 4096 4096 0.95 0.995 0.01
