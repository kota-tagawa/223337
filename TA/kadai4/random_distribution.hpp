#ifndef RANDOM_DISTRIBUTION_HPP
#define RANDOM_DISTRIBUTION_HPP

#include <vector>

// 標本平均を計算する関数
double calculateSampleMean(const std::vector<double>& data);

// 標本分散を計算する関数
double calculateSampleVariance(const std::vector<double>& data, double mean);

// 正規分布に従う乱数を生成する関数
std::vector<double> generateNormalDistribution(double mean, double variance, int N, int seed);

#endif // RANDOM_DISTRIBUTION_HPP