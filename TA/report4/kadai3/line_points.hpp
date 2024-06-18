#ifndef LINE_POINTS_HPP
#define LINE_POINTS_HPP

#include <iostream>
#include <vector>
#include <random>

class Vector2D {
private:
    double x;
    double y;

public:
    Vector2D(double _x = 0.0, double _y = 0.0) : x(_x), y(_y) {}

    double getX() const { return x; }
    double getY() const { return y; }
    void setX(double _x) { x = _x; } 
    void setY(double _y) { y = _y; } 

    // Vector2Dクラス表示用のオーバーライド
    friend std::ostream& operator<<(std::ostream& os, const Vector2D& vec) {
        os << vec.x << "," << vec.y;
        return os;
    }
};

std::vector<Vector2D> generateLinePoints(double A, double B, double C, int N);
std::vector<Vector2D> addGaussianNoise(std::vector<Vector2D>& points, double sigma, int seed);

#endif // LINE_POINTS_HPP
