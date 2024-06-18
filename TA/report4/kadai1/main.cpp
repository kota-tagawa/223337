// main.cpp

#include "random_distribution.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

using namespace std;

void kadai1(double mean, double variance, const vector<int>& sample_sizes, int seed){
    // 加点課題
    ofstream outfile("output.csv");
    if (!outfile) {
        cerr << "Failed to open file." << endl;
        return;
    }
    outfile << "N,SampleMean,SampleVariance,MeanDiffSquared,VarianceDiffSquared\n";

    for (int N : sample_sizes) {
        // 正規分布に従う乱数を生成
        vector<double> data = generateNormalDistribution(mean, variance, N, seed);

        // 標本平均と標本分散を計算
        double sampleMean = calculateSampleMean(data);
        double sampleVariance = calculateSampleVariance(data, sampleMean);

        // 平均と分散の差の2乗を計算
        double meanDiffSquared = (sampleMean - mean) * (sampleMean - mean);
        double varianceDiffSquared = (sampleVariance - variance) * (sampleVariance - variance);

        // 結果を出力
        cout << "Sample Mean (" << N << "):" << sampleMean << endl;
        cout << "Sample Variance (" << N << "):" <<sampleVariance << endl;

        // 加点課題
        outfile << N << "," << sampleMean << "," << sampleVariance << "," << meanDiffSquared << "," << varianceDiffSquared << "\n";
    }

}
int main(int argc, char* argv[]) {
    if (argc < 5) {
        cerr << "Usage: " << argv[0] << " <mean> <variance> <seed> <sample_sizes...>" << endl;
        return 1;
    }

    // 引数を取得
    double mean = stod(argv[1]);
    double variance = stod(argv[2]);
    int seed = stoi(argv[3]);
    // サンプルサイズを取得(加点課題)
    std::vector<int> sample_sizes;
    for (int i = 4; i < argc; i++) {
        sample_sizes.push_back(stoi(argv[i]));
    }
    kadai1(mean, variance, sample_sizes, seed);

    return 0;
}