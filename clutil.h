#ifndef _SIMPLEX_CL_UTILS_H
#define _SIMPLEX_CL_UTILS_H

#include <string>
#include <vector>
#include <CL/cl.h>

#include "main.h"


class CLSys {
public:
    ~CLSys();

    void setup();

    void showSystem() const;

    bool initDevice(unsigned platform, unsigned device);

    cl_program buildProgram(const std::string& code, const std::string& flags = "") const;
    cl_kernel createKernel(cl_program program, const std::string& main) const;
    cl_command_queue createQueue() const;

    cl_context ctx() const { return context; }
    cl_device_id device() const { return selectedDevId; }
    size_t deviceWorkGroupSize() const { return selectedDevWGsize; }

    void setDefaultCompilerFlags(const char* flags) { defaultFlags = flags; }

    static CLSys& getInstance() { return *instance; }
    static void destroy() { delete instance; }

private:
    struct clDevice {
        cl_device_id id;
        std::string name;
        size_t maxWorkGroup;
    };
    struct clPlatform {
        cl_platform_id id;
        std::string name;
        std::string version;
        std::vector<clDevice> devices;
    };

    CLSys();

    std::string defaultFlags;

    std::vector<clPlatform> platforms;
    cl_context context;
    cl_device_id selectedDevId;
    size_t selectedDevWGsize;

    void queryCLSystem();

    static CLSys* instance;
};

std::string clErrorString(cl_int error);

#endif
