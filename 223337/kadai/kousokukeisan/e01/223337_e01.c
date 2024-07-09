#include <stdio.h>
int main(){
    #pragma omp parallel for
    {
	for(int i=0;i<20;i++)
		printf("[i=%d]Hello World!\n",i);
    }
    return 0;
}
