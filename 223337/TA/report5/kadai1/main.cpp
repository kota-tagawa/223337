// main.cpp

#include <iostream>
#include "ReadDirectory.hpp"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <directory_path>" << std::endl;
        return 1;
    }

    std::string directory = argv[1];
    ReadDirectory reader(directory);
    reader.readFiles();

    const auto& files = reader.getFiles();
    if (files.empty()) {
        std::cout << "No files found in the directory." << std::endl;
    } else {
        std::cout << "Files in directory " << directory << ":" << std::endl;
        for (const auto& file : files) {
            std::cout << file << std::endl;
        }
    }
    return 0;
}
