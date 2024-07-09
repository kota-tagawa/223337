#include <iostream>
#include <dlfcn.h>

int main() {
    // ライブラリをロード
    void* handle = dlopen("./libcalc.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load library: " << dlerror() << std::endl;
        return 1;
    }

    // 関数のアドレスを取得
    auto my_add = reinterpret_cast<double(*)(double, double)> (dlsym(handle, "my_add"));
    auto my_sub = reinterpret_cast<double(*)(double, double)> (dlsym(handle, "my_sub"));

    // エラーチェック
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
        std::cerr << "Failed to load functions: " << dlsym_error << std::endl;
        dlclose(handle);
        return 1;
    }

    // 関数を使用
    double a = 5.0, b = 3.0;
    std::cout << "my_add(" << a << ", " << b << ") = " << my_add(a, b) << std::endl;
    std::cout << "my_sub(" << a << ", " << b << ") = " << my_sub(a, b) << std::endl;

    // ライブラリを解放
    dlclose(handle);

    return 0;
}
