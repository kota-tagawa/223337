#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv){

    int ierr;
    int myid, nprocs, len;
    char name[MPI_MAX_PROCESSOR_NAME];

    ierr = MPI_Init(&argc, &argv);

    ierr = MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    ierr = MPI_Get_processor_name(name, &len);

    if (myid == 0) {printf("Number of MPI Procceses: %d\n",nprocs);}
    printf("[HN: %s][Rank: %d]Hello world!\n",name ,myid);

    ierr = MPI_Finalize();
    
    return 0;
}