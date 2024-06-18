#include "RepeatMessage.hpp"
#include <iostream>

// kadai3
RepeatMessage::RepeatMessage(int n)
{
    nloops = n;
    message = nullptr;
}

std::ostream& operator<<(std::ostream &os, RepeatMessage& obj){
    int i = 0;
    while(i < obj.nloops){
        os << obj.getMessage();
        i++;
    }
    return os;
}