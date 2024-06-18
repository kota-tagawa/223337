#include <iostream>
#include "Message.hpp"

class RepeatMessage : public Message{
public:
    int nloops;
    RepeatMessage(int n);
};

std::ostream& operator<<(std::ostream &os, RepeatMessage& obj);