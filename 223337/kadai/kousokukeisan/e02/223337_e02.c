#include <stdio.h>
#include <time.h>
#include <omp.h>
#define def 1000000

int main(int argc, char *argv[])
{
    long int N;
    long int start,end,stride;
    long int number,factor;
    long int primecount;
    clock_t ts,te;
        int np;

    N = def;
    printf("Determining primes from 1 - %ld \n",N);

     omp_set_num_threads(4);

    stride = 2;
    start  = 3;
    end    = N;

    if (N==0 || N==1) primecount=0;
    if (N>=2) primecount=1;

    #pragma omp parallel shared(start,end,stride) private(number, factor) reduction(+: primecount)
        {
         ts = omp_get_wtime();
         #pragma omp for nowait schedule(dynamic, 3)
                 for(number=start; number<=end; number+=stride)
                                {
                factor = 3;
                while ((number%factor) != 0) factor+=2;
                if (factor == number){
                    primecount++;
                     }
                }
                printf("thread %d Elapsed Time: %f sec.\n", omp_get_thread_num(), omp_get_wtime()-ts);
    }

    printf("Found %ld primes\n",primecount);


    return 0;
}

