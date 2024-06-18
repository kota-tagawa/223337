#include <iostream>
#include "Message.h"
#include "Messages.h"
using namespace std;

// kadai2_1()
// コピーコンストラクタがないため、obj1とobj2は同じアドレス、メモリを参照する。
// obj1のデストラクタが呼ばれ、obj1のメモリが解放される。
// obj2のアドレスはそのまま同じメモリを指し示している多め、
// obj2のデストラクタが呼ばれ、既に解放されているメモリを再度解放しようとして、例外処理(コアダンプ)が発生する。
// そのため、コピーコンストラクタを作成し、アドレスをコピーするだけでなく、メモリを確保する必要がある。

void kadai2_2(){
    Message obj1("Hello World.");
    // Message obj2(obj1); //どちらでも可
    Message obj2 = obj1;
    cout << obj2 << endl;

    return;
}

void kadai2_3(){
    Messages msgs1("Hello");
    msgs1.addMessage("World");
    cout << msgs1.getMessage(0) << endl;
    cout << msgs1.getMessage(5) << endl;

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
    cout << "kadai2-2" << endl;
    kadai2_2();
    cout << endl;
    cout << "kadai2-3" << endl;
    kadai2_3();

    return 0;
}