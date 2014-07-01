#include "base.h"

namespace gpuip {

Base::Base(GpuEnvironment env, unsigned int width, unsigned int height)
        : _env(env), _w(width), _h(height)
{
    
}

bool Base::AddBuffer(const Buffer & buffer)
{
    if (_buffers.find(buffer.name) == _buffers.end()) {
        _buffers[buffer.name] = buffer;
        return true;
    } else {
        std::cerr << "gpuip error: Buffer named " << buffer.name
                  << " already exists. Skipping..." << std::endl;
        return false;
    }
}

Kernel * Base::CreateKernel(const std::string & name)
{
    _kernels.push_back(Kernel());
    _kernels.back().name = name;
    return &_kernels.back();
}

Kernel * Base::GetKernel(const std::string & name)
{
    for (int i = 0; i < _kernels.size(); ++i) {
        if (_kernels[i].name == name) {
            return &_kernels[i];
        }
    }
    std::cerr << "gpuip error: Could not find kernel named "
              << name << std::endl;
    return NULL;
}

} // end namespace gpuip
