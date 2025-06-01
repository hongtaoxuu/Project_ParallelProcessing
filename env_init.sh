#!/bin/bash
source /etc/profile.d/modules.sh && module load intel2025/compiler-rt/latest && module load intel2025/mkl/latest

export OMP_NUM_THREADS=30