#include "gpuip.h"
//----------------------------------------------------------------------------//
namespace gpuip {
//----------------------------------------------------------------------------//
Base * CreateOpenCL(unsigned int w, unsigned int h);
Base * CreateCUDA(unsigned int w, unsigned int h);
Base * CreateGLSL(unsigned int w, unsigned int h);
//----------------------------------------------------------------------------//
Base::Ptr Base::Create(GpuEnvironment env, unsigned int w, unsigned int h)
{
    switch(env) {
        case OpenCL:
            return Base::Ptr(CreateOpenCL(w,h));
        case CUDA:
            return Base::Ptr(CreateCUDA(w,h));
        case GLSL:
            return Base::Ptr(CreateGLSL(w,h));
        default:
            std::cerr << "gpuip error: Could not create env" << std::endl;
            return Base::Ptr();
    }
}
//----------------------------------------------------------------------------//
Base::Base(GpuEnvironment env, unsigned int width, unsigned int height)
        : _env(env), _w(width), _h(height)
{
    
}
//----------------------------------------------------------------------------//
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
//----------------------------------------------------------------------------//
Kernel::Ptr Base::CreateKernel(const std::string & name)
{
    _kernels.push_back(Kernel::Ptr(new Kernel()));
    _kernels.back()->name = name;
    return _kernels.back();
}
//----------------------------------------------------------------------------//
Kernel::Ptr Base::GetKernel(const std::string & name)
{
    for (int i = 0; i < _kernels.size(); ++i) {
        if (_kernels[i]->name == name) {
            return _kernels[i];
        }
    }
    std::cerr << "gpuip error: Could not find kernel named "
              << name << std::endl;
    return Kernel::Ptr();
}
//----------------------------------------------------------------------------//
} // end namespace gpuip
//----------------------------------------------------------------------------//
