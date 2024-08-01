#include <stdio.h>
#include <math.h>

#define N 5 // 行列のサイズ
#define EPSILON 1e-10 // 許容誤差
#define MAX_ITER 1000 // 最大反復回数

void matVecMul(double A[N][N], double x[N], double result[N]) {
    for (int i = 0; i < N; i++) {
        result[i] = 0.0;
        for (int j = 0; j < N; j++) {
            result[i] += A[i][j] * x[j];
        }
    }
}

double vecDot(double x[N], double y[N]) {
    double result = 0.0;
    for (int i = 0; i < N; i++) {
        result += x[i] * y[i];
    }
    return result;
}

void vecAdd(double x[N], double y[N], double result[N]) {
    for (int i = 0; i < N; i++) {
        result[i] = x[i] + y[i];
    }
}

void vecSub(double x[N], double y[N], double result[N]) {
    for (int i = 0; i < N; i++) {
        result[i] = x[i] - y[i];
    }
}

void vecScale(double x[N], double scalar, double result[N]) {
    for (int i = 0; i < N; i++) {
        result[i] = x[i] * scalar;
    }
}

void conjugateGradientMethod(double A[N][N], double b[N], double x[N]) {
    double r[N], p[N], Ap[N];
    double alpha, beta, rsold, rsnew;
    int iteration = 0;

    // 初期化 r = b - Ax, p = r
    matVecMul(A, x, Ap);
    vecSub(b, Ap, r);
    for (int i = 0; i < N; i++) {
        p[i] = r[i];
    }

    rsold = vecDot(r, r);

    printf("初期値:\n");
    for (int i = 0; i < N; i++) {
        printf("x[%d] = %lf\n", i, x[i]);
    }

    while (sqrt(rsold) > EPSILON && iteration < MAX_ITER) {
        matVecMul(A, p, Ap);
        alpha = rsold / vecDot(p, Ap);
        
        // x = x + alpha * p
        for (int i = 0; i < N; i++) {
            x[i] = x[i] + alpha * p[i];
        }

        // r = r - alpha * Ap
        for (int i = 0; i < N; i++) {
            r[i] = r[i] - alpha * Ap[i];
        }

        rsnew = vecDot(r, r);
        if (sqrt(rsnew) < EPSILON) {
            break;
        }

        beta = rsnew / rsold;

        // p = r + beta * p
        for (int i = 0; i < N; i++) {
            p[i] = r[i] + beta * p[i];
        }

        rsold = rsnew;
        iteration++;
    }

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

    conjugateGradientMethod(A, b, x);

    return 0;
}
