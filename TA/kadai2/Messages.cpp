#include <iostream>
#include "Messages.h"

// コンストラクタ
Messages::Messages(){
    messages;
}
// 入力する文字列が1つの場合のコンストラクタ
Messages::Messages(const std::string& message_string){
    messages.push_back(message_string);
}
// 入力する文字列が複数ある場合のコンストラクタ
Messages::Messages(const std::vector<std::string>& message_vector){
    messages = message_vector;
}

// デストラクタ
Messages::~Messages(){
    if(!messages.empty()) messages.clear();
}

// 文字列を追加する関数
void Messages::addMessage(const std::string& message_string){
    messages.push_back(message_string);
}

// vectorリストのインデックスを指定して、該当する文字列を返す関数
std::string Messages::getMessage(int message_id){
    std::string str = "";
    if (message_id < messages.size()) {
        str = messages[message_id];
    } else {
        std::cerr << "Error: Index out of range" << std::endl;
        exit(1);
    }
    return str;
}

// すべての文字列を表示する関数
void Messages::showAllMessages(void){
    for(int i = 0; i < messages.size(); ++i){
        std::cout << messages[i] << " ";
    }
    std::cout << std::endl;
}

// 文字列の数を返す関数
int Messages::getNMessages(void){
    int NMessages = messages.size();

    return NMessages;
}