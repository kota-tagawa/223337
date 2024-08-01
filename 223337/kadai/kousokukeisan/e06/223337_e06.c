#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 1024*1024*200/sizeof(int)

void bcast(int *data, int rank, int size) {
    double start, end;
    int i;
    // synchronize
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    MPI_Bcast(data, N, MPI_INT, 0, MPI_COMM_WORLD);

    end = MPI_Wtime();
    if (rank == 0) {
        printf("MPI_Bcast: %f seconds\n", end - start);
    }
}

void send_recv(int *data, int rank, int size) {
    double start, end;
    int i;
    // synchronize
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    if (rank == 0) {
        for (i = 1; i < size; i++) {
            MPI_Send(data, N, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(data, N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    end = MPI_Wtime();
    if (rank == 0) {
        printf("MPI_Send/MPI_Recv: %f seconds\n", end - start);
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int *data = (int*)malloc(N * sizeof(int));
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize data
    int i;
    if (rank == 0) {
        for (i = 0; i < N; i++) {
            data[i] = i;
        }
    }

    // use bcast
    bcast(data, rank, size);
    // use send and recv
    send_recv(data, rank, size);

    free(data);
    MPI_Finalize();
    return 0;
}
