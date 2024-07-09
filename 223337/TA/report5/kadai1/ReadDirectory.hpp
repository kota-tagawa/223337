// ReadDirectory.h

#ifndef READDIRECTORY_H
#define READDIRECTORY_H

#include <vector>
#include <string>

class ReadDirectory{
public:
    ReadDirectory(const std::string& directory);
    void readFiles();
    const std::vector<std::string>& getFiles() const;

private:
    std::string directory;
    std::vector<std::string> files;
};

#endif // READDIRECTORY_H
