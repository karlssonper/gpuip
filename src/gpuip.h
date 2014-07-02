#ifndef GPUIP_H_
#define GPUIP_H_

#include <string>
#include <map>
#include <vector>
#include <iostream>

namespace gpuip {

enum GpuEnvironment { OpenCL, CUDA, GLSL };

class Factory
{
  public:
    static class Base * Create(GpuEnvironment env,
                               unsigned int width,
                               unsigned int height);
};
       
struct Buffer {
    enum CopyOperation{ READ_DATA, WRITE_DATA };
    std::string name;
    unsigned int channels;
    unsigned int bpp; // bytes per pixel
};

template<typename T>
struct Parameter
{
    std::string name;
    T value;
};

struct Kernel {
    std::string name;
    std::string code;
    std::vector<std::string> inBuffers;
    std::vector<std::string> outBuffers;
    std::vector<Parameter<int> > paramsInt;
    std::vector<Parameter<float> > paramsFloat;
};

} //end namespace gpuip

#endif
