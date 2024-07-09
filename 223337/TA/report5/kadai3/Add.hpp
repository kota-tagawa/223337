#ifndef ADD_H
#define ADD_H

#include "plugin.hpp"
#include <string>
#include <memory>

class Add : public plugin {
public:
    Add() {}
    ~Add() {}
    std::string getPluginName (void) override {
        return "plugin : Add";
    }
    double exec(double a, double b) override {
        return a + b;
    }
};
#endif // ADD_H