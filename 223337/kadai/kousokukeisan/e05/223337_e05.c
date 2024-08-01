#include <stdio.h>
#include <mpi.h>
#include <time.h>
#define def 1000000

int main(int argc, char **argv) {
    long int N;
    long int start, end;
    long int number;
    long int primecount;
    long int tmp;
    int factor;
    clock_t ts, te;
    int ierr;
    int myid, nprocs;
    MPI_Status istatus;

    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    N = def;
    if (myid == 0) {
        printf("Determining primes from 1 - %ld \n", N);
    }

    ts = clock();

    tmp = 0;
    if (myid == 0 && N >= 2) tmp = 1;

    for (number = myid * 2 + 3; number <= N; number += nprocs * 2) {
        factor = 3;
        while ((number % factor) != 0 && factor * factor <= number) {
            factor += 2;
        }
        if (factor * factor > number) {
            tmp++;
        }
    }

    ierr = MPI_Reduce(&tmp, &primecount, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    te = clock();
    if (myid == 0) {
        printf("Total primes found: %ld\n", primecount);
        printf("Total MPI processes: %d\n", nprocs);
        printf("Elapsed Time: %f sec.\n", (float)(te - ts) / CLOCKS_PER_SEC);
    }

    ierr = MPI_Finalize();
    return 0;
}
