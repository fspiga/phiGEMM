#!/bin/bash

export NUMA_CTL="numactl -m 1 -c 1"
export OMP_NUM_THREADS=6 
export MKL_NUM_THREADS=6
export CUDA_VISIBLE_DEVICES=0

#export EXE=single_test-zgemm.x
export EXE=single_test-dgemm.x

if [ -z "$1" ]; then
cat  << 'EOF' > .list_tests
N N 4096 4096 4096
EOF
export TEST_FILENAME=.list_tests
else
export TEST_FILENAME=$1
fi

while read -r x
do
	env LD_LIBRARY_PATH=../lib:${LD_LIBRARY_PATH} ${NUMA_CTL} ../bin/${EXE} 1 ${x} 0.95 0.995 0.01
done < ${TEST_FILENAME}

rm -f .list_tests
