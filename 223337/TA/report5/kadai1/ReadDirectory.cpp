// ReadDirectory.cpp

#include "ReadDirectory.hpp"
#include <dirent.h>
#include <string.h>
#include <iostream>

ReadDirectory::ReadDirectory(const std::string& dir) : directory(dir) {}

void ReadDirectory::readFiles() {
    files.clear();
    DIR* dir = opendir(directory.c_str());
    if (dir) {
        struct dirent* file;
        while ((file = readdir(dir)) != nullptr) {
            // Check if it's a regular file
            if (strcmp (file->d_name, ".") != 0 && strcmp (file->d_name, "..") != 0)  { 
                files.push_back(file->d_name);
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Error opening directory: " << directory << std::endl;
    }
}

const std::vector<std::string>& ReadDirectory::getFiles() const {
    return files;
}