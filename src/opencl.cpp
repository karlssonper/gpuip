#include "opencl.h"
//----------------------------------------------------------------------------//
namespace gpuip {
//----------------------------------------------------------------------------//
Base *
CreateOpenCL(unsigned int width,
                    unsigned int height)
{
    return new OpenCLImpl(width, height);
}
//----------------------------------------------------------------------------//
OpenCLImpl::OpenCLImpl(unsigned int width,
                       unsigned int height)
        : Base(OpenCL, width, height)
{
    cl_int err;

    // Get Platform ID
    cl_platform_id platform_id;
    err = clGetPlatformIDs(1, &platform_id, NULL);
    // Check error TODO

    // TODO add option for CPU? Could be nice for testing
    cl_device_type device_type = CL_DEVICE_TYPE_GPU;

    // Get Device ID
    err = clGetDeviceIDs(platform_id, device_type,
                         1 /*only 1 device for now */,
                         &_device_id, NULL);
    // Check error TODO

    // Create context and command queue
    _ctx = clCreateContext(NULL, 1, &_device_id, NULL, NULL, NULL);
    _queue = clCreateCommandQueue(_ctx, _device_id, 0, NULL);
}
//----------------------------------------------------------------------------//
bool
OpenCLImpl::InitBuffers(std::string * error)
{
    cl_int err;
    std::map<std::string,Buffer>::const_iterator it;
    for (it = _buffers.begin(); it != _buffers.end(); ++it) {
        _clBuffers[it->second.name] = clCreateBuffer(
            _ctx, CL_MEM_READ_WRITE /* TODO optimize for only read/write */,
            _GetBufferSize(it->second), NULL, &err);
        //TODO check error
    }
    return true;
}
//----------------------------------------------------------------------------//
bool
OpenCLImpl::Build(std::string * error)
{
    cl_int err;

    for (int i = 0; i < _kernels.size(); ++i) {
        const char * code = _kernels[i].code.c_str();
        cl_program program = clCreateProgramWithSource(
            _ctx, 1, &code, NULL,  &err);
        // check error TODO

        // Build program
        err = clBuildProgram(program, 1, &_device_id, NULL, NULL, NULL);
        // check error

        // Create kernel from program
        const char * kernel_name = _kernels[i].name.c_str();
        _clKernels.push_back(clCreateKernel(program, kernel_name, &err));
        // check ERROR todo
    }
    return true;
}
//----------------------------------------------------------------------------//
bool
OpenCLImpl::Process(std::string * err)
{
    for(int i = 0; i < _kernels.size(); ++i) {
        if (!_EnqueueKernel(_kernels[i], _clKernels[i], err)) {
            return false;
        }
    }
    return true;
}
//----------------------------------------------------------------------------//
bool
OpenCLImpl::Copy(const std::string & buffer,
                 Buffer::CopyOperation op,
                 void * data,
                 std::string * err)
{
    cl_int cl_err;
    if (op == Buffer::READ_DATA) {
        cl_err =  clEnqueueReadBuffer(
            _queue,  _clBuffers[buffer],
            CL_TRUE /* function call returns when copy is done */ ,
            0, _GetBufferSize(_buffers[buffer]), data, 0 , NULL, NULL);
    } else if (op == Buffer::WRITE_DATA) {
        cl_err =  clEnqueueWriteBuffer(
            _queue,  _clBuffers[buffer],
            CL_TRUE /* function call returns when copy is done */ ,
            0, _GetBufferSize(_buffers[buffer]), data, 0 , NULL, NULL);
    }
    // TODO check error
    
    return true;
}
//----------------------------------------------------------------------------//
bool OpenCLImpl::_EnqueueKernel(const Kernel & kernel,
                                const cl_kernel & clKernel,
                                std::string * err)
{
    cl_int cl_err;
    cl_int argc = 0;
    
    // Set kernel arguments in the following order:
    // 1. Input buffers.
    const size_t size = sizeof(cl_mem);
    for (int j = 0; j < kernel.inBuffers.size(); ++j) {
        cl_err = clSetKernelArg(clKernel, argc++, size,
                                &_clBuffers[kernel.inBuffers[j]]);
    }

    // 2. Output buffers.
    for (int j = 0; j < kernel.outBuffers.size(); ++j) {
        cl_err = clSetKernelArg(clKernel, argc++, size,
                                &_clBuffers[kernel.outBuffers[j]]);
    }

    // 3. Int parameters
    for(int i = 0; i < kernel.paramsInt.size(); ++i) {
        cl_err = clSetKernelArg(clKernel, argc++, sizeof(int),
                             &kernel.paramsInt[i].value);
    }

    // 4. Float parameters
    for(float i = 0; i < kernel.paramsFloat.size(); ++i) {
        cl_err = clSetKernelArg(clKernel, argc++, sizeof(float),
                             &kernel.paramsFloat[i].value);
    }

    const size_t global_work_size[] = { _w, _h };    
    cl_err = clEnqueueNDRangeKernel(_queue, clKernel, 2, NULL,
                                 global_work_size, NULL, 0, NULL, NULL);
    //TODO check error
    
    return true;
}
//----------------------------------------------------------------------------//
} // end namespace gpuip
