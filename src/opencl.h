#ifndef GPUIP_OPENCL_H_
#define GPUIP_OPENCL_H_

#include "base.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace gpuip {

class OpenCLImpl : public Base
{
  public:
    OpenCLImpl(unsigned int width, unsigned int height);

    virtual bool InitBuffers(std::string * err);
    
    virtual bool Build(std::string * err);

    virtual bool Process(std::string * err);

    virtual bool Copy(const std::string & buffer,
                      Buffer::CopyOperation op,
                      void * data,
                      std::string * err);
    
  protected:
    cl_device_id _device_id;
    cl_context _ctx;
    cl_command_queue _queue;

    std::vector<cl_kernel> _clKernels;
    std::map<std::string, cl_mem> _clBuffers;

    bool _EnqueueKernel(const Kernel & kernel,
                        const cl_kernel & clKernel,
                        std::string * err);
};

} // end namespace gpuip

#endif
