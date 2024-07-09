# include "FileLister.h"
# include <dirent.h>
# include <string.h>

FileLister::FileLister(std::string _DirectoryPath){
    DirectoryPath = _DirectoryPath;
}

std::vector<std::string> FileLister::getFileNames(){
    DIR* dir = opendir (DirectoryPath.c_str());
    struct dirent* file;
    while ((file = readdir (dir)) != nullptr) {
        if (strcmp (file->d_name, ".") != 0 && strcmp (file->d_name, "..") != 0) {
            FileNames.push_back(file->d_name);
        }
    }
    closedir(dir);
    return FileNames;
}