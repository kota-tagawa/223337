#include <stdio.h>
#include <math.h>

#define N 5 // 行列のサイズ
#define EPSILON 1e-10 // 許容誤差
#define MAX_ITER 1000 // 最大反復回数

void gaussSeidelMethod(double A[N][N], double b[N], double x[N]) {
    double x_new[N];
    int iteration = 0;
    double error;

    // 初期値の表示
    printf("初期値:\n");
    for (int i = 0; i < N; i++) {
        printf("x[%d] = %lf\n", i, x[i]);
    }

    do {
        error = 0.0;
        for (int i = 0; i < N; i++) {
            x_new[i] = b[i];
            for (int j = 0; j < N; j++) {
                if (i != j) {
                    x_new[i] -= A[i][j] * x[j];
                }
            }
            x_new[i] /= A[i][i];

            // 誤差の計算
            error = fmax(error, fabs(x_new[i] - x[i]));
            x[i] = x_new[i];
        }

        iteration++;
    } while (error > EPSILON && iteration < MAX_ITER);

    // 結果の表示
    printf("収束後の解:\n");
    for (int i = 0; i < N; i++) {
        printf("x[%d] = %lf\n", i, x[i]);
    }
    printf("反復回数: %d\n", iteration);
}

int main() {
    // 行列 A とベクトル b
    double A[N][N] = {
        {3.0, 1.0, 0.0, 0.0, 0.0},
        {1.0, 2.0, 1.0, 0.0, 0.0},
        {0.0, 1.0, 1.0, 1.0, 0.0},
        {0.0, 0.0, 1.0, 4.0, 1.0},
        {0.0, 0.0, 0.0, 1.0, 5.0}
    };

    double b[N] = {7.0, 9.0, 9.0, 21.0, 23.0};
    double x[N] = {0, 0, 0, 0, 0}; // 初期値

    gaussSeidelMethod(A, b, x);

    return 0;
}
