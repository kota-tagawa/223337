#!/bin/bash
#PBS -q wEduq
#PBS -l select=1:ncpus=1
#PBS -v DOCKER_IMAGE=prg-env:latest
ulimit -c 0          # avoid generating core files
source /etc/profile  # enable module command
. /common/intel-2022/compiler/latest/env/vars.sh  # Intel compiler
. /common/intel-2022/mpi/latest/env/vars.sh       # Intel MPI
. /common/intel-2022/mkl/latest/env/vars.sh       # Intel MKL
export OMP_NUM_THREADS=1
#echo "Current directory is " `pwd`
cd $PBS_O_WORKDIR
#echo "Current directory is " `pwd`
icx kadai3.c -o kadai3.x
./kadai3.x