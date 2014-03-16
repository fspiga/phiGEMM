#!/bin/bash

NUMA_CTL="numactl -m 1 -c 1"

echo "'\nTesting DGEMM\n"
make TEST_DATATYPE_FLAGS=-D__CUDA_TYPE_DOUBLE
env LD_LIBRARY_PATH=../lib:${LD_LIBRARY_PATH} ${NUMA_CTL} ../bin/single_test.x 1 4096 4096 4096 0.6 0.91 0.1

sleep 1

echo "'\nTesting ZGEMM\n"
make TEST_DATATYPE_FLAGS=-D__CUDA_TYPE_DOUBLE_COMPLEX
env LD_LIBRARY_PATH=../lib:${LD_LIBRARY_PATH} ${NUMA_CTL} ../bin/single_test.x 1 4096 4096 4096 0.6 0.91 0.1
