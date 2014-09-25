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
#include "implinterface.h"
#include <sstream>
#ifdef _WIN32
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif
//----------------------------------------------------------------------------//
namespace gpuip {
//----------------------------------------------------------------------------//
Buffer::Buffer(const std::string & name, PixelType type, unsigned int channels)
        : _name(name), _type(type), _channels(_channels)
{
}
//----------------------------------------------------------------------------//
void Buffer::SetType(PixelType type)
{
    _type = type;
}
//----------------------------------------------------------------------------//
void Buffer::SetChannels(unsigned int channels)
{
    _channels = channels;
}
//----------------------------------------------------------------------------//
Kernel::BufferLink::BufferLink(const std::string & name, Buffer::Ptr buffer)
        : _name(name), _buffer(buffer)
{
}
//----------------------------------------------------------------------------//
Kernel::Kernel(const std::string & name)
        : _name(name)
{
}
//----------------------------------------------------------------------------//
void Kernel::SetCode(const std::string & code)
{
    _code = code;
}
//----------------------------------------------------------------------------//
void Kernel::AddInputBuffer(const std::string & name, Buffer::Ptr buffer)
{
    _inputBuffers.push_back(BufferLink(name, buffer));
}
//----------------------------------------------------------------------------//
void Kernel::AddOutputBuffer(const std::string & name, Buffer::Ptr buffer)
{
    _outputBuffers.push_back(BufferLink(name, buffer));
}
//----------------------------------------------------------------------------//
template<typename T>
void _SetParam(std::vector<Parameter<T> > & params,
               const std::string & name,
               T value)
{
    for(size_t i = 0; i < params.size(); ++i) {
        if(params[i].Name() == name) {
            params[i].SetValue(value);
            return;
        }
    }
    params.push_back(Parameter<T>(name, value));
}
void Kernel::SetParamInt(const std::string & name, int value)
{
    _SetParam(_paramsInt, name, value);
}
void Kernel::SetParamFloat(const std::string & name, float value)
{
    _SetParam(_paramsFloat, name, value);
}
//----------------------------------------------------------------------------//
inline const char * _GetSharedLibraryFilename(GpuEnvironment env)
{
    std::stringstream ss;
#ifndef _WIN32
    ss << "lib";
#endif
    const char * envStr[3] = { "OpenCL", "CUDA", "GLSL" };
    ss << "gpuip" << envStr[env];
#ifdef _WIN32
    ss << ".dll";
#else
#  ifdef __APPLE__
    ss << ".dylib";
#  else
    ss << ".so"; // linux
#  endif
#endif
    return ss.str().c_str();
}
//----------------------------------------------------------------------------//
ImageProcessor::Ptr ImageProcessor::Create(GpuEnvironment env)
{
    return ImageProcessor::Ptr(
        new ImageProcessor(_GetSharedLibraryFilename(env)));
}
//----------------------------------------------------------------------------//
ImageProcessor::Ptr ImageProcessor::Create(const std::string & sharedLibrary)
{
    return ImageProcessor::Ptr(new ImageProcessor(sharedLibrary.c_str()));
}
//----------------------------------------------------------------------------//
bool ImageProcessor::CanCreate(GpuEnvironment env)
{
    return ImageProcessor::CanCreate(_GetSharedLibraryFilename(env));
}
//----------------------------------------------------------------------------//
bool ImageProcessor::CanCreate(const std::string & filename)
{
    /*
    void * handle = dlopen(filename.c_str(), RTLD_NOW);
    if (handle == NULL) {
        return false;
    }

    if (dlsym(handle, "CreateImpl")) {
        std::cerr << dlerror() << std::endl;
    }
    
    if (handle != NULL && dlsym(handle, "CreateImpl") != NULL) {
        dlclose(handle);
        return true;
    } */
    return false;
}
//----------------------------------------------------------------------------//
ImageProcessor::ImageProcessor(const char * filename)
        : _dynamicLibObj(NULL), _impl(NULL)
{
    // Open shared library
#ifdef _WIN32
    _dynamicLibObj = LoadLibrary(filename);
#else
    _dynamicLibObj = dlopen(filename, RTLD_NOW);
#endif
    if (_dynamicLibObj == NULL) {
        std::stringstream ss;
        ss << "Could not open shared library " << filename << "\n";
#ifdef _WIN32
        ss << GetLastError();
#else
        const char * err = dlerror();
        if (err) {
            ss << err;
        }
#endif
        throw std::runtime_error(ss.str().c_str());
    }
    /*
    // Look for CreateImpl symbol
    void* loadSym = dlsym(_dynamicLibObj, "CreateImpl");
    if(loadSym == NULL) {
        std::stringstream ss;
        ss << "No CreateImpl function in shared library " << filename << "\n";
        const char * err = dlerror();
        if (err) {
            ss << err;
        }
        throw std::runtime_error(ss.str().c_str());
    }

    // Create implementation object
    _impl = reinterpret_cast<CreateImplFunc>(loadSym)();
    if (_impl == NULL) {
        std::stringstream ss;
        ss << "Could not create gpuip impl from shared library " << filename;
        throw std::runtime_error(ss.str().c_str());
    }
    */
}
//----------------------------------------------------------------------------//
ImageProcessor::~ImageProcessor()
{
    /*
    // Look for CreateImpl symbol
    void* loadSym = dlsym(_dynamicLibObj, "DeleteImpl");
    if(loadSym == NULL) {
        throw std::runtime_error("No DeleteImpl function in shared library");
    }

    // Delete implementation
    reinterpret_cast<DeleteImplFunc>(loadSym)(_impl);

    // Close shared library
    dlclose(_dynamicLibObj);
    */
}
//----------------------------------------------------------------------------//
Buffer::Ptr ImageProcessor::CreateBuffer(const std::string & name,
                                         Buffer::PixelType type,
                                         unsigned int channels)
{
    return _impl->CreateBuffer(name, type, channels);
}
//----------------------------------------------------------------------------//
Kernel::Ptr ImageProcessor::CreateKernel(const std::string & name)
{
    return _impl->CreateKernel(name);
}
//----------------------------------------------------------------------------//
void ImageProcessor::SetDimensions(unsigned int width, unsigned int height)
{
    _impl->SetDimensions(width, height);
}
//----------------------------------------------------------------------------//
double ImageProcessor::Allocate(std::string * error)
{
    return _impl->Allocate(error);
}
//----------------------------------------------------------------------------//
double ImageProcessor::Build(std::string * error)
{
    return _impl->Build(error);
}
//----------------------------------------------------------------------------//
double ImageProcessor::Run(std::string * error)
{
    return _impl->Run(error);
}
//----------------------------------------------------------------------------//
double ImageProcessor::Copy(Buffer::Ptr buffer,
                            Buffer::CopyOperation operation,
                            void * data,
                            std::string * error)
{
    return _impl->Copy(buffer, operation, data, error);
}
//----------------------------------------------------------------------------//
std::string ImageProcessor::BoilerplateCode(Kernel::Ptr kernel) const
{
    return _impl->BoilerplateCode(kernel);
}
//----------------------------------------------------------------------------//
} // end namespace gpuip
//----------------------------------------------------------------------------//
