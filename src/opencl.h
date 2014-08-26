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

#ifndef GPUIP_OPENCL_H_
#define GPUIP_OPENCL_H_
//----------------------------------------------------------------------------//
#include "gpuip.h"
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
//----------------------------------------------------------------------------//
namespace gpuip {
//----------------------------------------------------------------------------//
class OpenCLImpl : public ImageProcessor
{
  public:
    OpenCLImpl();

    virtual ~OpenCLImpl();

    virtual double Allocate(std::string * err);
    
    virtual double Build(std::string * err);

    virtual double Run(std::string * err);

    virtual double Copy(Buffer::Ptr buffer,
                        Buffer::CopyOperation op,
                        void * data,
                        std::string * err);

    virtual std::string BoilerplateCode(Kernel::Ptr kernel) const;
    
  protected:
    cl_device_id _device_id;
    cl_context _ctx;
    cl_command_queue _queue;

    std::vector<cl_kernel> _clKernels;
    std::map<std::string, cl_mem> _clBuffers;

  private:
    bool _EnqueueKernel(const Kernel & kernel,
                        const cl_kernel & clKernel,
                        cl_event & event,
                        std::string * err);

    bool _ReleaseBuffers(std::string * err);

    bool _ReleaseKernels(std::string * err);
};
//----------------------------------------------------------------------------//
} // end namespace gpuip
//----------------------------------------------------------------------------//
#endif
