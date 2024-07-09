# include <vector>
# include <string>

class FileLister{
    public:
        FileLister(std::string _DirectoryPath);
        std::vector<std::string> getFileNames();

    private:
        std::string DirectoryPath;
        std::vector<std::string> FileNames;
};