#include <iostream >
class Message {
protected:
    char* message;

public:
Message();
Message(const char* _message);
Message(const Message& obj);    //コピーコンストラクタ
~Message();
void setMessage (const char* _message);
char* getMessage (void);
};
std::istream &operator >>(std::istream& stream, Message& obj);
std::ostream &operator <<(std::ostream& stream, Message& obj);