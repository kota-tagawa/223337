#include <iostream>
#include "linear_algebra.hpp"

using namespace std;
void kadai2(){
    // ベクトルと行列の初期化
    Eigen::Vector3d a(1, 2, 3);
    Eigen::Vector3d b(4, 5, 6);
    Eigen::Matrix3d A;
    A << 1, 2, 3,
        4, 5, 6,
        7, 8, 9;
    Eigen::Matrix3d B;
    B << 9, 8, 7,
        6, 5, 4,
        3, 2, 1;

    // ベクトル同士の和 a + b
    Eigen::Vector3d sum = addVectors(a, b);
    std::cout << "a + b = \n" << sum << std::endl << std::endl;

    // ベクトル同士の積 ab^⊤ (外積)
    Eigen::Matrix3d outer = outerProduct(a, b);
    std::cout << "ab^T = \n" << outer << std::endl << std::endl;

    // ベクトルの内積 a^⊤b
    double inner = innerProduct(a, b);
    std::cout << "a^Tb = \n" << inner << std::endl << std::endl;

    // 行列とベクトルの積 Ab
    Eigen::Vector3d matVec = matrixVectorProduct(A, b);
    cout << "Ab = \n" << matVec << endl << endl;

    // 行列同士の積 AB
    Eigen::Matrix3d matMat = matrixMatrixProduct(A, B);
    cout << "AB = \n" << matMat << endl;
}

int main(int argc, char* argv[]){
    kadai2();
    return 0;
}
