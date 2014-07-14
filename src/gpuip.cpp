#include "gpuip.h"
//----------------------------------------------------------------------------//
namespace gpuip {
//----------------------------------------------------------------------------//
Base * CreateOpenCL();
Base * CreateCUDA();
Base * CreateGLSL();
//----------------------------------------------------------------------------//
Base::Ptr Base::Create(GpuEnvironment env)
{
    switch(env) {
        case OpenCL:
            return Base::Ptr(CreateOpenCL());
        case CUDA:
            return Base::Ptr(CreateCUDA());
        case GLSL:
            return Base::Ptr(CreateGLSL());
        default:
            std::cerr << "gpuip error: Could not create env" << std::endl;
            return Base::Ptr();
    }
}
//----------------------------------------------------------------------------//
Base::Base(GpuEnvironment env)
        : _env(env), _w(0), _h(0)
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
void Base::SetDimensions(unsigned int width, unsigned int height)
{
    _w = width;
    _h = height;
}
//----------------------------------------------------------------------------//
} // end namespace gpuip
//----------------------------------------------------------------------------//
