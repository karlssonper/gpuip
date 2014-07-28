#include "gpuip.h"
//----------------------------------------------------------------------------//
namespace gpuip {
//----------------------------------------------------------------------------//
#ifdef _GPUIP_OPENCL
Base * CreateOpenCL();
#endif
//----------------------------------------------------------------------------//
#ifdef _GPUIP_CUDA
Base * CreateCUDA();
#endif
//----------------------------------------------------------------------------//
#ifdef _GPUIP_GLSL
Base * CreateGLSL();
#endif
//----------------------------------------------------------------------------//
Base::Ptr Base::Create(GpuEnvironment env)
{
    switch(env) {
        case OpenCL:
#ifdef _GPUIP_OPENCL
            return Base::Ptr(CreateOpenCL());
#else
            throw std::logic_error("gpuip was not built with OpenCL");
#endif
        case CUDA:
#ifdef _GPUIP_CUDA
            return Base::Ptr(CreateCUDA());
#else
            throw std::logic_error("gpuip was not built with CUDA");
#endif
        case GLSL:
#ifdef _GPUIP_GLSL
            return Base::Ptr(CreateGLSL());
#else
            throw std::logic_error("gpuip was not built with GLSL");
#endif
        default:
            std::cerr << "gpuip error: Could not create env" << std::endl;
            return Base::Ptr();
    }
}
//----------------------------------------------------------------------------//
bool Base::CanCreateGpuEnvironment(GpuEnvironment env)
{
    switch(env) {
        case OpenCL:
#ifdef _GPUIP_OPENCL
            return true;
#else
            return false;
#endif
        case CUDA:
#ifdef _GPUIP_CUDA
            return true;
#else
            return false;
#endif
        case GLSL:
#ifdef _GPUIP_GLSL
            return true;
#else
            return false;
#endif
        default:
            return false;
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
    for (size_t i = 0; i < _kernels.size(); ++i) {
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
unsigned int  Base::_GetBufferSize(const Buffer & buffer) const
{
    unsigned int bpp = buffer.bpp;

    // Special case in OpenCL since it pads with {uchar,float}4
    // even if the array is of type {uchar,float}3
    if ((_env == OpenCL or _env == CUDA) and buffer.channels == 3) {
        bpp = (bpp /3 ) * 4;
    }
        
    return bpp * _w * _h;
}
//----------------------------------------------------------------------------//
} // end namespace gpuip
//----------------------------------------------------------------------------//
