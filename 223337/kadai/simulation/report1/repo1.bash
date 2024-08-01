#!/bin/bash
#PBS -q wEduq
#PBS -l select=1:ncpus=4
#PBS -v DOCKER_IMAGE=prg-env:latest
ulimit -c 0
source /etc/profile
source /common/intel-2022/setvars.sh &> /dev/null
export OMP_NUM_THREADS=4
cd $PBS_O_WORKDIR
icx -qopenmp himenoBMTxpa.c -o himenoBMTxpa.x
./himenoBMTxpa.x S