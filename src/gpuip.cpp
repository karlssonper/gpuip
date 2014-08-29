/*
The MIT License (MIT)

Copyright (c) 2014 Per Karlsson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "gpuip.h"
//----------------------------------------------------------------------------//
#ifdef _GPUIP_OPENCL
#include "opencl.h"
#endif
//----------------------------------------------------------------------------//
#ifdef _GPUIP_CUDA
#include "cuda.h"
#endif
//----------------------------------------------------------------------------//
#ifdef _GPUIP_GLSL
#include "glsl.h"
#endif
//----------------------------------------------------------------------------//
namespace gpuip {
//----------------------------------------------------------------------------//
ImageProcessor::Ptr ImageProcessor::Create(GpuEnvironment env)
{
    switch(env) {
        case OpenCL:
#ifdef _GPUIP_OPENCL
            return ImageProcessor::Ptr(new OpenCLImpl());
#else
            throw std::logic_error("gpuip was not built with OpenCL");
#endif
        case CUDA:
#ifdef _GPUIP_CUDA
            return ImageProcessor::Ptr(new CUDAImpl());
#else
            throw std::logic_error("gpuip was not built with CUDA");
#endif
        case GLSL:
#ifdef _GPUIP_GLSL
            return ImageProcessor::Ptr(new GLSLImpl());
#else
            throw std::logic_error("gpuip was not built with GLSL");
#endif
        default:
            std::cerr << "gpuip error: Could not create env" << std::endl;
            return ImageProcessor::Ptr();
    }
}
//----------------------------------------------------------------------------//
bool ImageProcessor::CanCreate(GpuEnvironment env)
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
Buffer::Buffer(const std::string & name_, Type type_, unsigned int channels_)
        : name(name_), type(type_), channels(channels_)
{
}
//----------------------------------------------------------------------------//
Kernel::Kernel(const std::string & name_)
        : name(name_)
{
}
//----------------------------------------------------------------------------//
Kernel::BufferLink::BufferLink(Buffer::Ptr buffer_, const std::string & name_)
        : buffer(buffer_), name(name_)
{
}
//----------------------------------------------------------------------------//
ImageProcessor::ImageProcessor(GpuEnvironment env)
        : _env(env), _w(0), _h(0)
{
    
}
//----------------------------------------------------------------------------//
Buffer::Ptr
ImageProcessor::CreateBuffer(const std::string & name,
                             Buffer::Type type,
                             unsigned int channels)
{
    if (_buffers.find(name) == _buffers.end()) {
        Buffer::Ptr p = Buffer::Ptr(new Buffer(name, type, channels));
        _buffers[name] = p;
        return p;
    } else {
        std::cerr << "gpuip error: Buffer named " << name
                  << " already exists. Skipping..." << std::endl;
        return Buffer::Ptr(new Buffer(name, type, channels));
    }
}
//----------------------------------------------------------------------------//
Kernel::Ptr ImageProcessor::CreateKernel(const std::string & name)
{
    _kernels.push_back(Kernel::Ptr(new Kernel(name)));
    return _kernels.back();
}
//----------------------------------------------------------------------------//
void ImageProcessor::SetDimensions(unsigned int width, unsigned int height)
{
    _w = width;
    _h = height;
}
//----------------------------------------------------------------------------//
double ImageProcessor::Allocate(std::string * error)
{
    throw std::logic_error("'Allocate' not implemented in subclass");
}
//----------------------------------------------------------------------------//
double ImageProcessor::Build(std::string * error)
{
    throw std::logic_error("'Build' not implemented in subclass");
}
//----------------------------------------------------------------------------//
double ImageProcessor::Run(std::string * error)
{
    throw std::logic_error("'Run' not implemented in subclass");
}
//----------------------------------------------------------------------------//
double ImageProcessor::Copy(Buffer::Ptr buffer,
                            Buffer::CopyOperation operation,
                            void * data,
                            std::string * error)
{
    throw std::logic_error("'Copy' not implemented in subclass");
}
//----------------------------------------------------------------------------//
std::string ImageProcessor::BoilerplateCode(Kernel::Ptr kernel) const
{
    throw std::logic_error("'BoilerplateCode' not implemented in subclass");
}
//----------------------------------------------------------------------------//
unsigned int  ImageProcessor::_BufferSize(Buffer::Ptr buffer) const
{
    unsigned int bpp = 0; // bytes per pixel
    switch(buffer->type) {
        case Buffer::UNSIGNED_BYTE:
            bpp = buffer->channels;
            break;
        case Buffer::HALF:
            bpp = sizeof(float)/2 * buffer->channels;
            break;
        case Buffer::FLOAT:
            bpp = sizeof(float) * buffer->channels;
            break;
    }
    return bpp * _w * _h;
}
//----------------------------------------------------------------------------//
} // end namespace gpuip
//----------------------------------------------------------------------------//
