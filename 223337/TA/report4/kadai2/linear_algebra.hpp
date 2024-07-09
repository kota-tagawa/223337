#ifndef LINEAR_ALGEBRA_HPP
#define LINEAR_ALGEBRA_HPP

#include <Eigen/Dense>

Eigen::Vector3d addVectors(const Eigen::Vector3d& a, const Eigen::Vector3d& b);
Eigen::Matrix3d outerProduct(const Eigen::Vector3d& a, const Eigen::Vector3d& b);
double innerProduct(const Eigen::Vector3d& a, const Eigen::Vector3d& b);
Eigen::Vector3d matrixVectorProduct(const Eigen::Matrix3d& A, const Eigen::Vector3d& b);
Eigen::Matrix3d matrixMatrixProduct(const Eigen::Matrix3d& A, const Eigen::Matrix3d& B);

#endif // LINEAR_ALGEBRA_HPP
