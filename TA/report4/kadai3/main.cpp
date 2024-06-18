#include <iostream>
#include <fstream>

#include "line_points.hpp"

using namespace std;

void kadai3() {
    // 加点課題
    ofstream outfile("output.csv");
    if (!outfile) {
        cerr << "Failed to open file." << endl;
        return;
    }
    outfile << "x,y,noisy_x,noisy_y\n";

    // 直線のパラメータ
    double A = 1.0;
    double B = -1.0;
    double C = 0.0;

    // データ点の数と誤差の分散
    int N = 100;
    double sigma = 1.0;
    int seed = 1234;

    // 直線上の点を生成
    auto points = generateLinePoints(A, B, C, N);

    // 誤差を付加
    auto noisy_points = addGaussianNoise(points, sigma, seed);

    // 結果を出力
    for (size_t i = 0; i < points.size(); i++) {
        outfile << points[i] << "," << noisy_points[i] << "\n";
        cout << points[i] << "," << noisy_points[i] << endl;
    }
}
int main(int argc, char* argv[]){
    kadai3();
    return 0;
}