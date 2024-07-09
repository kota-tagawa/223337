#include <stdio.h>

int main(){

    int n = 4;
    double alpha = 1.0;
    double beta = 0.0;

    double A[4][4] = {
        {1.0, 2.0, 3.0, 4.0},{8.0, 7.0, 6.0, 5.0},
        {9.0, 10.0, 11.0, 12.0},{16.0, 15.0, 14.0, 13.0}
    };
    double B[4][4] = {
        {0.1, 0.2, 0.3, 0.4},{0.8, 0.7, 0.6, 0.5},
        {0.9, 1.0, 1.1, 1.2},{1.6, 1.5, 1.4, 1.3}
    };
    double C[4][4] = {0};

    // 行列積計算
    dgemm("N","N",&n,&n,&n,&alpha,A,&n,&B,&n,&beta,C,&n);

    for (int i = 0; i < 4; i++){
        for (int j = 0; j < 4; j++){
            printf("C[%d][%d]:%f, ",i,j,C[i][j]);
        }
        printf("\n");
    }
    return 0;
}