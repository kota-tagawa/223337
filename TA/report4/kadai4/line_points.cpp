#include "line_points.hpp"

// 直線上の点を生成
std::vector<Vector2D> generateLinePoints(double A, double B, double C, int N) {
    std::vector<Vector2D> points;
    for (int alpha = 0; alpha < N; alpha++) {
        double x_alpha = -10 + 20.0 * alpha / (N - 1);
        double y_alpha = -(A * x_alpha + C) / B;
        points.emplace_back(x_alpha, y_alpha);
    }
    return points;
}

// 正規分布に従う誤差を付加
std::vector<Vector2D> addGaussianNoise(std::vector<Vector2D>& points, double sigma, int seed) {
    std::mt19937 method(seed);
    std::normal_distribution<double> distribution(0.0, sigma);

    for (auto& point : points) {
        point.setX(point.getX() + distribution(method));
        point.setY(point.getY() + distribution(method));
    }
    return points;
}

// 最小二乗法によって関数を推定
std::vector<double> leastSquaresFit(const std::vector<Vector2D>& points) {
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_xy = 0.0;
    double sum_x_squared = 0.0;
    int N = points.size();

    for (const auto& point : points) {
        sum_x += point.getX();
        sum_y += point.getY();
        sum_xy += point.getX() * point.getY();
        sum_x_squared += point.getX() * point.getY();
    }

    double m = (N * sum_xy - sum_x * sum_y) / (N * sum_x_squared - sum_x * sum_x);
    double c = (sum_y - m * sum_x) / N;

    return {m, c};
}