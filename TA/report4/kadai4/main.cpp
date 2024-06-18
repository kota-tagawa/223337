#include <iostream>
#include <fstream>

#include "line_points.hpp"

using namespace std;

void kadai4() {
    // 直線のパラメータ
    double A = 1.0;
    double B = -1.0;
    double C = 1.0;

    // データ点の数と誤差の分散
    int N = 100;
    double sigma = 1.0;
    int seed = 1234;

    // 直線上の点を生成
    auto points = generateLinePoints(A, B, C, N);

    // 誤差を付加
    auto noisy_points = addGaussianNoise(points, sigma, seed);

    // 最小二乗法
    vector<double> params = leastSquaresFit(noisy_points);

    // 結果を出力
    cout << "Result:" << endl;
    cout << "Slope (m): " << params[0] << endl;
    cout << "Y-intercept (c): " << params[1] << endl;

}
int main(int argc, char* argv[]){
    kadai4();
    return 0;
}