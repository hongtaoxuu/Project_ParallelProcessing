#!/bin/bash
source /etc/profile.d/modules.sh
module load intel2025/compiler-rt/latest
module load intel2025/mkl/latest

# export MKL_NUM_THREADS=10
export OMP_NUM_THREADS=30

./dgemm_x86 19

