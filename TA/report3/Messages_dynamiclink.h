#ifndef MESSAGES_H
#define MESSAGES_H

#include <string>
#include <vector>

#ifdef _WIN32
    #ifdef MESSAGES_EXPORTS
        #define MESSAGES_API __declspec(dllexport)
    #else
        #define MESSAGES_API __declspec(dllimport)
    #endif
#else
    #define MESSAGES_API
#endif

class MESSAGES_API Messages {
private:
    std::vector<std::string> messages;

public:
    Messages();
    Messages(const std::string& message_string);
    Messages(const std::vector<std::string>& message_vector);
    ~Messages();

    void addMessage(const std::string& message_string);
    std::string getMessage(int message_id);
    void showAllMessages();
    int getNMessages();
};

#endif // MESSAGES_H
