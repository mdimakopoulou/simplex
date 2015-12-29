#include <iostream>

#include "clutil.h"

using namespace std;


string clErrorString(cl_int error) {
    switch(error) {
        case CL_SUCCESS:                            return "Success!";
        case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
        case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:                   return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
        case CL_MAP_FAILURE:                        return "Map failure";
        case CL_INVALID_VALUE:                      return "Invalid value";
        case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
        case CL_INVALID_PLATFORM:                   return "Invalid platform";
        case CL_INVALID_DEVICE:                     return "Invalid device";
        case CL_INVALID_CONTEXT:                    return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
        case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
        case CL_INVALID_SAMPLER:                    return "Invalid sampler";
        case CL_INVALID_BINARY:                     return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
        case CL_INVALID_PROGRAM:                    return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
        case CL_INVALID_KERNEL:                     return "Invalid kernel";
        case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
        case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
        case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
        case CL_INVALID_EVENT:                      return "Invalid event";
        case CL_INVALID_OPERATION:                  return "Invalid operation";
        case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
        default: return "Unknown";
    }
}

string getPlatformParam(cl_platform_id id, cl_platform_info param) {
    size_t size = 0;
    clGetPlatformInfo(id, param, 0, nullptr, &size);

    string result;
    result.resize(size);
    clGetPlatformInfo(id, param, size, const_cast<char*> (result.data ()), nullptr);

    return result;
}

string getPlatformVersion(cl_platform_id id) {
    return getPlatformParam(id, CL_PLATFORM_VERSION);
}

string getPlatformName(cl_platform_id id) {
    return getPlatformParam(id, CL_PLATFORM_NAME);
}

string getDeviceName(cl_device_id id) {
    size_t size = 0;
    clGetDeviceInfo(id, CL_DEVICE_NAME, 0, nullptr, &size);

    string result;
    result.resize(size);
    clGetDeviceInfo(id, CL_DEVICE_NAME, size, const_cast<char*> (result.data ()), nullptr);

    return result;
}

bool handleCLError(cl_int error) {
    if (error == CL_SUCCESS) {
        return true;
    }

    cerr << "CL error: (" << -error << ") " << clErrorString(error) << endl;
    return false;
}


CLSys* CLSys::instance = new CLSys();

CLSys::CLSys()
    : defaultFlags(""), context(0)
{
}

CLSys::~CLSys()
{
    if (verbose) {
        cout << "CL: releasing resources\n";
    }
    if (!context) {
        clReleaseContext(context);
    }
}

void CLSys::setup() {
    queryCLSystem();
}

void CLSys::showSystem() const {
    int pIdx = 0;
    cout << "Total: " << platforms.size() << " platforms\n"; 
    for (clPlatform p : platforms) {
        cout << "  Platform[" << pIdx << "] (" << p.devices.size() << " devices) [V: " << p.version << "]\t::" << p.name << endl;

        int dIdx = 0;
        for (clDevice d : p.devices) {
            cout << "    Device[" << dIdx << "] :: " << d.name << "\t| W/G max = " << d.maxWorkGroup << endl;
            ++dIdx;
        }

        ++pIdx;
    }
}

bool CLSys::initDevice(unsigned platform, unsigned device) {
    if (platform < 0 || platform >= platforms.size()) {
        cerr << "bad platform selector: " << platform << endl;
        return false;
    }

    clPlatform& selectedPlatform = platforms[platform];
    if (verbose) {
        cout << "CL: selected platform: [" << selectedPlatform.name << "] Version: " << selectedPlatform.version << "\n";
    }

    if (device < 0 || device >= selectedPlatform.devices.size()) {
        cerr << "bad device selector: " << device << endl;
        return false;
    }

    clDevice& selectedDev = selectedPlatform.devices[device];
    if (verbose) {
        cout << "CL: selected device: [" << selectedDev.name << "]\n";
    }
    selectedDevId = selectedDev.id;
    selectedDevWGsize = selectedDev.maxWorkGroup;

    const cl_context_properties ctxProperties [] = {
        CL_CONTEXT_PLATFORM,
        reinterpret_cast<cl_context_properties>(selectedPlatform.id),
        0, 0
    };
    cl_int error = CL_SUCCESS;
    context = clCreateContext(ctxProperties, 1, &selectedDev.id, nullptr, nullptr, &error);
    if (!handleCLError(error)) {
        return false;
    }
    if (verbose) {
        cout << "CL: context created\n";
    }

    return true;
}

void showCompilerLog(cl_program program, cl_device_id id) {
    size_t size = 0;
    clGetProgramBuildInfo(program, id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
    char* buf = new char[size+1];
    clGetProgramBuildInfo(program, id, CL_PROGRAM_BUILD_LOG, size, buf, nullptr);
    char *end = buf + size;
    *end = 0;
    while (end > buf) {
        if (*end) {
            if (isspace(*end)) *end = 0;
            else break;
        }
        --end;
    }
    cerr << "Program build log:\n" << buf << "\n-----\n";
    delete [] buf;
}

cl_program CLSys::buildProgram(const string& code, const string& flags) const {
    size_t length[1] = {code.size()};
    const char* source[1] = {code.data()};

    cl_int error = CL_SUCCESS;
    cl_program program = clCreateProgramWithSource(context, 1, source, length, &error);
    if (!handleCLError(error)) {
        return 0;
    }

    const char* cflags = flags.c_str();
    if (cflags[0] == 0 && defaultFlags.length() > 0) {
        cflags = defaultFlags.c_str();
    }

    cl_device_id ids[1] = {selectedDevId};
    error = clBuildProgram(program, 1, ids, flags.c_str(), nullptr, nullptr);
    if (!handleCLError(error)) {
        showCompilerLog(program, ids[0]);
        return 0;
    }

    if (verbose) {
        showCompilerLog(program, ids[0]);
    }
    return program;
}

cl_kernel CLSys::createKernel(cl_program program, const std::string& main) const {
    cl_int error = CL_SUCCESS;
    cl_kernel kernel = clCreateKernel(program, main.c_str(), &error);
    if (!handleCLError(error)) {
        return 0;
    }

    return kernel;
}

cl_command_queue CLSys::createQueue() const {
    cl_int error = CL_SUCCESS;
    cl_command_queue q = clCreateCommandQueueWithProperties(context, selectedDevId, 0, &error);
    if (!handleCLError(error)) {
        return 0;
    }
    
    return q;
}

void CLSys::queryCLSystem() {
    cl_uint platformIdCount = 0;
    clGetPlatformIDs(0, nullptr, &platformIdCount);

    vector<cl_platform_id> platformIds(platformIdCount);
    clGetPlatformIDs(platformIdCount, platformIds.data(), nullptr);

    for (cl_uint p = 0; p < platformIdCount; ++p) {
        clPlatform platform;
        platform.id = platformIds[p];
        platform.name = getPlatformName(platform.id);
        platform.version = getPlatformVersion(platform.id);

        cl_uint deviceIdCount = 0;
        clGetDeviceIDs(platformIds[p], CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceIdCount);

        vector<cl_device_id> deviceIds(deviceIdCount);
        clGetDeviceIDs(platformIds[p], CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds.data(), nullptr);

        for (cl_uint d = 0; d < deviceIdCount; ++d) {
            clDevice device;
            device.id = deviceIds[d];
            device.name = getDeviceName(device.id);

            clGetDeviceInfo(device.id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(device.maxWorkGroup), &device.maxWorkGroup, nullptr);

            platform.devices.push_back(device);
        }

        platforms.push_back(platform);
    }
}
