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
#include <set>
#include <algorithm>
#include <sstream>
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
Buffer::Buffer(const std::string & name_,
               Type type_,
               unsigned int width_,
               unsigned int height_,
               unsigned int channels_)
        : name(name_), type(type_), width(width_), height(height_),
          channels(channels_), isTexture(false)
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
        : _env(env)
{
    
}
//----------------------------------------------------------------------------//
Buffer::Ptr
ImageProcessor::CreateBuffer(const std::string & name,
                             Buffer::Type type,
                             unsigned int width,
                             unsigned int height,
                             unsigned int channels)
{
    if (_buffers.find(name) == _buffers.end()) {
        Buffer::Ptr p = Buffer::Ptr(
            new Buffer(name, type, width, height, channels));
        _buffers[name] = p;
        return p;
    } else {
        std::cerr << "gpuip error: Buffer named " << name
                  << " already exists. Skipping..." << std::endl;
        return Buffer::Ptr(new Buffer(name, type, width, height, channels));
    }
}
//----------------------------------------------------------------------------//
Kernel::Ptr ImageProcessor::CreateKernel(const std::string & name)
{
    _kernels.push_back(Kernel::Ptr(new Kernel(name)));
    return _kernels.back();
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
    return bpp * buffer->width * buffer->height;
}
//----------------------------------------------------------------------------//
ImageProcessor::_BufferReadWriteType ImageProcessor::_GetBufferReadWriteType(
    Buffer::Ptr buf)
{
    bool read = false;
    bool write = false; 
    for(size_t i = 0; i < _kernels.size(); ++i) {
        for(size_t j = 0; j < _kernels[i]->inBuffers.size(); ++j) {
            if(_kernels[i]->inBuffers[j].buffer->name == buf->name) {
                read = true;
            }
        }
        for(size_t j = 0; j < _kernels[i]->outBuffers.size(); ++j) {
            if(_kernels[i]->outBuffers[j].buffer->name == buf->name) {
                write = true;
            }
        }
    }
    
    if(!read && !write) {
        return BUFFER_NOT_USED;
    } else if (read && !write) {
        return BUFFER_READ_ONLY;
    } else if (!read && write) {
        return BUFFER_WRITE_ONLY;
    } else {
        return BUFFER_READ_AND_WRITE;
    }
}
//----------------------------------------------------------------------------//
bool ImageProcessor::_ValidateBuffers(std::string * error)
{
    std::map<std::string,Buffer::Ptr>::const_iterator it;
    for(it = _buffers.begin(); it != _buffers.end(); ++it) {
        const Buffer::Ptr & buffer = it->second;
        if(buffer->name.empty()) {
            (*error) += "Buffer name empty.\n";
            return false;
        }
        if(!buffer->width) {
            (*error) += buffer->name;
            (*error) += " has width = 0.\n";
            return false;
        }
        if(!buffer->height) {
            (*error) += buffer->name;
            (*error) += " has height = 0.\n";
            return false;
        }
        if(buffer->channels < 1 || buffer->channels > 4) {
            (*error) += buffer->name;
            (*error) += " has channels outside range [1,4].\n";
            return false;
        }
    }
    return true;
}
//----------------------------------------------------------------------------//
bool ImageProcessor::_ValidateKernels(std::string * err)
{
    std::stringstream ss;
    ss << "gpuip error. kernel ";
    for(size_t i = 0; i < _kernels.size(); ++i) {
        const Kernel::Ptr & kernel = _kernels[i];
        if(kernel->name.empty()) {
            ss << "name empty.\n";
            (*err) += ss.str();
            return false;
        }
        if(kernel->code.empty()) {
            ss << kernel->name << " code empty.\n";
            (*err) += ss.str();
            return false;
        }

        if(kernel->outBuffers.empty()) {
            ss << kernel->name << " has no output buffer(s).\n";
            (*err) += ss.str();
            return false;
        }
        
        std::set<std::string> inBuffers;
        std::set<std::string> inBuffersName;
        for(size_t j = 0; j < kernel->inBuffers.size(); ++j) {
            if(!inBuffers.insert(kernel->inBuffers[j].buffer->name).second) {
                ss << kernel->name << ": "
                   << kernel->inBuffers[j].buffer->name
                   << " used twice as input buffer.\n";
                (*err) += ss.str();
                return false;
            }
            if(!inBuffersName.insert(kernel->inBuffers[j].name).second) {
                ss << kernel->name << ": "
                   << kernel->inBuffers[j].name
                   << " used twice as naming of input buffer.\n";
                (*err) += ss.str();
                return false;
            }
        }
        
        std::set<std::string> outBuffers;
        std::set<std::string> outBuffersName;
        for(size_t j = 0; j < kernel->outBuffers.size(); ++j) {
            if(!outBuffers.insert(kernel->outBuffers[j].buffer->name).second) {
                ss << kernel->name << ": "
                   << kernel->outBuffers[j].buffer->name
                   << " used twice as output buffer.\n";
                (*err) += ss.str();
                return false;
            }
            if(!outBuffersName.insert(kernel->outBuffers[j].name).second) {
                ss << kernel->name << ": "
                   << kernel->outBuffers[j].name
                   << " used twice as naming of output buffer.\n";
                (*err) += ss.str();
                return false;
            }
        }
        
        std::set<std::string> buffersIntersect;
        std::set_intersection(
            inBuffers.begin(), inBuffers.end(),
            outBuffers.begin(), outBuffers.end(),
            std::inserter(buffersIntersect, buffersIntersect.begin()));
        if(!buffersIntersect.empty()) {
            ss << kernel->name << ": "
               << *buffersIntersect.begin()
               << " used as both input and output buffer.\n";
            (*err) += ss.str();
            return false;
        }
        
        std::set<std::string> nameIntersect;
        std::set_intersection(
            inBuffersName.begin(), inBuffersName.end(),
            outBuffersName.begin(), outBuffersName.end(),
            std::inserter(nameIntersect, nameIntersect.begin()));
        if(!nameIntersect.empty()) {
            ss << kernel->name << ": "
               << *nameIntersect.begin()
               << " used as both input and output naming of buffers.\n";
            (*err) += ss.str();
            return false;
        }
        
        const unsigned int width = kernel->outBuffers.front().buffer->width;
        const unsigned int height = kernel->outBuffers.front().buffer->height;
        for(size_t j = 1; j < kernel->outBuffers.size(); ++j) {
            if(kernel->outBuffers[j].buffer->width != width) {
                ss << kernel->name << ": output buffer "
                   << kernel->outBuffers[j].buffer->name
                   << " does not have the same width as the 1st output buffer "
                   << kernel->outBuffers.front().buffer->name
                   << ". " << kernel->outBuffers[j].buffer->width 
                   << " vs " <<  width << ".\n";
                (*err) += ss.str();
                return false;
            }
            if(kernel->outBuffers[j].buffer->height != height) {
                ss << kernel->name << ": output buffer "
                   << kernel->outBuffers[j].buffer->name
                   << " does not have the same height as the 1st output buffer "
                   << kernel->outBuffers[j].buffer->name
                   << ". " << kernel->outBuffers[j].buffer->height 
                   << " vs " << height << ".\n";
                (*err) += ss.str();
                return false;
            }
        }
      
        std::set<std::string> paramsName;
        for(size_t j = 0; j < kernel->paramsInt.size(); ++j) {
            if(!paramsName.insert(kernel->paramsInt[j].name).second) {
                ss << kernel->name << ": parameter name "
                   << kernel->paramsInt[j].name << " used more than once.\n";
                (*err) += ss.str();
                return false;
            }
        }
        for(size_t j = 0; j < kernel->paramsFloat.size(); ++j) {
            if(!paramsName.insert(kernel->paramsFloat[j].name).second) {
                ss << kernel->name << ": parameter name "
                   << kernel->paramsFloat[j].name << " used more than once.\n";
                (*err) += ss.str();
                return false;
            }
        }
    }
    return true;
}
//----------------------------------------------------------------------------//
} // end namespace gpuip
//----------------------------------------------------------------------------//
