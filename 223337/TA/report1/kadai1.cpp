#include <iostream>
#include "RepeatMessage.hpp"

// kadai1
void kadai1(){
    Message obj1;
    obj1.setMessage("Hello World.");
    std::cout << obj1.getMessage() << std::endl;
}
//kadai2
void kadai2(){
    Message obj2;
    std::cout << "Input message: ";
    std::cin >> obj2;
    std::cout << "Output message: ";
    std::cout << obj2 << std::endl;
}

//kadai3
void kadai3(){
    RepeatMessage obj3(3);
    std::cout << "input message: ";
    std::cin >> obj3;
    std::cout << "Output message: ";
    std::cout << obj3 << std::endl;
}

int main(int argc, char *argv[]){
    kadai1();
    kadai2();
    kadai3();
    return 0;
}