#include <dlfcn.h>
#include <iostream>
#include "PluginInterface.h"
#include "FileLister.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <directory_path>" << std::endl;
        return 1;
    }

    std::string directoryPath = argv[1];
    FileLister fl(directoryPath);

    std::vector<std::string> fileNames = fl.getFileNames();

    for (const auto& fileName : fileNames) {
        std::string fullpath = directoryPath + "/" + fileName.c_str();
        void* handle = dlopen(fullpath.c_str(), RTLD_LAZY);
        if (!handle) {
            std::cerr << "Failed to load library: " << dlerror() << std::endl;
            return 1;
        }
        auto new_instance = reinterpret_cast<std::unique_ptr<PluginInterface>(*)(void)>(dlsym (handle, "new_instance"));

        auto instance = new_instance();
        std::cout << instance->getPluginName() << std::endl;
        std::cout << instance->exec(10, 5) << std::endl;
    }

    return 0;
}