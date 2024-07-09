#ifndef __PLUGIN_H__
#define __PLUGIN_H__

#include <string>
#include <memory>

class plugin {
public:
    virtual ~plugin() = default;
    virtual std::string getPluginName (void) = 0;
    virtual double exec (double a, double b) = 0;
};

#endif // __PLUGIN_H__