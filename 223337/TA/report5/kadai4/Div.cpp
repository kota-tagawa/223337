#include <string>
#include "PluginInterface.h"

class Div: public PluginInterface{
public:
    std::string getPluginName (void) override{return "PluginName:Div";}
    double exec(double a, double b) override{return a / b;}
};

extern "C" {
    std::unique_ptr<PluginInterface> new_instance (void) {
        return std::unique_ptr<PluginInterface> (new Div);
    }
}