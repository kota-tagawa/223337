#include "Message.hpp"
#include <iostream>
#include <cstring>

// kadai1
void Message::setMessage(const char *m){
    message = new char[std::strlen(m) + 1];
    std::strcpy(message, m);
}
const char* Message::getMessage(){
    return message;
}

// kadai2
std::istream& operator>>(std::istream &is, Message& obj){
    std::string buffer;
    std::getline(is, buffer);
    obj.setMessage(buffer.c_str());
    return is;
}
std::ostream& operator<<(std::ostream &os, Message& obj){
    os << obj.getMessage();
    return os;
}