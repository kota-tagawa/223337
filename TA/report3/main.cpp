#include <iostream>
//#include "Messages.h"
#include "Messages_dynamiclink.h"
using namespace std;


void kadai3(){
    Messages msgs1("Hello");
    msgs1.addMessage("World");
    cout << msgs1.getMessage(0) << endl;
    cout << msgs1.getMessage(1) << endl;

    Messages msgs2;
    msgs2.addMessage("A");
    msgs2.addMessage("Whole");
    msgs2.addMessage("New");
    msgs2.addMessage("World");
    cout << msgs2.getNMessages() << endl;
    msgs2.showAllMessages();

    std::vector<std::string> msgs = {"apple", "banana", "cake", "donut"};
    Messages msgs3(msgs);
    msgs3.addMessage("egg");
    cout << msgs3.getMessage(4) << endl;
    cout << msgs3.getNMessages() << endl;

    return;
}

int main(int argc, char* argv[]){
    cout << "kadai3" << endl;
    kadai3();

    return 0;
}