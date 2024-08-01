#!/bin/bash
#PBS -q wEduq
#PBS -l select=1:ncpus=4:mpiprocs=4
#PBS -v DOCKER_IMAGE=prg-env:latest,DOCKER_OPTIONS="--network=overlaynw"

ulimit -c 0
source /etc/profile
source /common/intel-2022/setvars.sh &> /dev/null
source /common/intel-2022/mkl/latest/env/vars.sh &> /dev/null

module load intel
module load intelmpi.intel

cd $PBS_O_WORKDIR
mpiicc 223337_e04.c -o e04.x
mpirun -np 4 ./e04.x