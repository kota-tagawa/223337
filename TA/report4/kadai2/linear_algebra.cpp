#include "linear_algebra.hpp"

// ベクトル同士の和 a + b
Eigen::Vector3d addVectors(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    return a + b;
}

// ベクトル同士の積 ab^⊤ (外積)
Eigen::Matrix3d outerProduct(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    return a * b.transpose();
}

// ベクトルの内積 a^⊤b
double innerProduct(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    return a.dot(b);
}

// 行列とベクトルの積 Ab
Eigen::Vector3d matrixVectorProduct(const Eigen::Matrix3d& A, const Eigen::Vector3d& b) {
    return A * b;
}

// 行列同士の積 AB
Eigen::Matrix3d matrixMatrixProduct(const Eigen::Matrix3d& A, const Eigen::Matrix3d& B) {
    return A * B;
}
