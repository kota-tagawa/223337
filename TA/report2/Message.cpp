#include <stdlib.h>
#include <string.h>
#include <string>
#include "Message.h"

Message::Message(){
    message = NULL;
}

//コンストラクタ
Message::Message(const char* _message){
    message = new char[strlen(_message) + 1];
    strcpy(message, _message);
}

//コピーコンストラクタ（クラスと同名でconst付きで参照）
//コンストラクタと同様に、メモリを確保
//メモリを確保しない場合は、デフォルトコピーコンストラクタでよい
Message::Message(const Message& obj){
    message = new char[strlen(obj.message) + 1];
    strcpy(message, obj.message);
}

//デストラクタ
Message::~Message(){
    if(message != NULL) delete[] message;
}

//セッター
void Message::setMessage(const char* _message){
   if(message) delete[] message;
    message = new char [strlen(_message) + 1];
    strcpy (message, _message);
}
//ゲッター
char* Message::getMessage (void){
    return message;
}
//演算子のオーバーライド
std::istream& operator >>(std::istream& stream, Message& obj) {
    std::string buffer;
    std::getline(stream, buffer);
    obj.setMessage(buffer.c_str());

    return stream;
}
std::ostream& operator <<(std::ostream& stream, Message& obj) {
    stream << obj.getMessage();
    return stream;
}