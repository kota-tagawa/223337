#include "Add.hpp"
#include <memory>

extern "C" {
    std::unique_ptr<plugin> new_instance (void) {
        return std::unique_ptr<plugin> (new Add);
    }
}