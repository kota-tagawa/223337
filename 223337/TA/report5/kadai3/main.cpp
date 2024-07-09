#include <iostream>
#include <dlfcn.h>
#include <memory>
#include "plugin.hpp"

int main() {
    // ライブラリをロード
    void* handle = dlopen("./libaddplugin.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load library: " << dlerror() << std::endl;
        return 1;
    }

    // 関数のアドレスを取得
    auto new_instance = reinterpret_cast<std::unique_ptr<plugin>(*)()>(dlsym(handle, "new_instance"));
    if (!new_instance) {
        const char* dlsym_error = dlerror();
        std::cerr << "Failed to load function new_instance: " << dlsym_error << std::endl;
        return 1;
    }
    std::unique_ptr<plugin>instance = new_instance();

    // 関数を使用
    double a = 5.0, b = 3.0;
    std::cout << "Plugin Name: " << instance->getPluginName() << std::endl;
    std::cout << "Result of " << a << " + " << b << " = " << instance->exec(a, b) << std::endl;

    // ライブラリを解放
    return 0;
}
