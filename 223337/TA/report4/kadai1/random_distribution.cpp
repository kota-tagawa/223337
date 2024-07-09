#include "random_distribution.hpp"
#include <random>
#include <numeric>
#include <cmath>

// 標本平均を計算する関数
double calculateSampleMean(const std::vector<double>& data) {
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
}

// 標本分散を計算する関数
double calculateSampleVariance(const std::vector<double>& data, double mean) {
    double variance = 0.0;
    for (double value : data) {
        variance += (value - mean) * (value - mean);
    }
    return variance / (data.size() - 1);
}

// 正規分布に従う乱数を生成する関数
std::vector<double> generateNormalDistribution(double mean, double variance, int N, int seed) {
    double stddev = std::sqrt(variance); // 標準偏差
    std::mt19937 method(seed); // 乱数生成器初期化
    std::normal_distribution<double> dist(mean, stddev); // 正規乱数を生成
    std::vector<double> data(N); 
    for (int i = 0; i < N; ++i) {
        data[i] = dist(method);
    }
    return data;
}