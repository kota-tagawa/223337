#include <iostream>

// kadai1
class Message{
protected:
    char *message;
public:
    void setMessage(const char *m);
    const char* getMessage();
};

// kadai2
std::istream& operator>>(std::istream &is, Message& obj);
std::ostream& operator<<(std::ostream &os, Message& obj);