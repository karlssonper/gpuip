#include "opencl.h"
#include "opencl_error.h"
//----------------------------------------------------------------------------//
namespace gpuip {
//----------------------------------------------------------------------------//
Base *
CreateOpenCL()
{
    return new OpenCLImpl();
}
//----------------------------------------------------------------------------//
OpenCLImpl::OpenCLImpl()
        : Base(OpenCL)
{
    // Get Platform ID
    cl_platform_id platform_id;
    if(clGetPlatformIDs(1, &platform_id, NULL) != CL_SUCCESS) {
        throw std::logic_error("gpuip::OpenCLImpl() could not get platform id");
    }
    
    // TODO add option for CPU? Could be nice for testing
    cl_device_type device_type = CL_DEVICE_TYPE_GPU;

    // Get Device ID
    if(clGetDeviceIDs(platform_id, device_type,
                      1 /*only 1 device for now */,
                      &_device_id, NULL) != CL_SUCCESS) {
        throw std::logic_error("gpuip::OpenCLImpl() could not get device id");
    }
    
    // Create context and command queue
    _ctx = clCreateContext(NULL, 1, &_device_id, NULL, NULL, NULL);
    _queue = clCreateCommandQueue(_ctx, _device_id, 0, NULL);
}
//----------------------------------------------------------------------------//
bool
OpenCLImpl::InitBuffers(std::string * err)
{
    cl_int cl_err;
    std::map<std::string,Buffer>::const_iterator it;
    for (it = _buffers.begin(); it != _buffers.end(); ++it) {
        _clBuffers[it->second.name] = clCreateBuffer(
            _ctx, CL_MEM_READ_WRITE /* TODO optimize for only read/write */,
            _GetBufferSize(it->second), NULL, &cl_err);
        if (_clErrorInitBuffers(cl_err, err)) {
            return false;
        }
    }
    return true;
}
//----------------------------------------------------------------------------//
bool
OpenCLImpl::Build(std::string * error)
{
    cl_int cl_err;

    for (int i = 0; i < _kernels.size(); ++i) {
        const char * code = _kernels[i]->code.c_str();
        const char * name = _kernels[i]->name.c_str();
        cl_program program = clCreateProgramWithSource(
            _ctx, 1, &code, NULL,  &cl_err);
        if (_clErrorCreateProgram(cl_err, error)) {
            return false;
        }
        
        // Build program
        cl_err = clBuildProgram(program, 1, &_device_id, NULL, NULL, NULL);
        if (_clErrorBuildProgram(cl_err, error, program, _device_id, name)) {
            return false;
        }
    
        // Create kernel from program
        _clKernels.push_back(clCreateKernel(program, name, &cl_err));
        if (_clErrorCreateKernel(cl_err, error)) {
            return false;
        }
    }
    return true;
}
//----------------------------------------------------------------------------//
bool
OpenCLImpl::Process(std::string * err)
{
    for(int i = 0; i < _kernels.size(); ++i) {
        if (!_EnqueueKernel(*_kernels[i].get(), _clKernels[i], err)) {
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
                 std::string * error)
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
    if (_clErrorCopy(cl_err, error, buffer, op)) {
        return false;
    }
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

    // It should be fine to check once all the arguments have been set
    if (_clErrorSetKernelArg(cl_err, err, kernel.name)) {
        return false;
    }
    
    const size_t global_work_size[] = { _w, _h };    
    cl_err = clEnqueueNDRangeKernel(_queue, clKernel, 2, NULL,
                                    global_work_size, NULL, 0, NULL, NULL);

    if (_clErrorEnqueueKernel(cl_err, err, kernel)) {
        return false;
    }
        
    return true;
}
//----------------------------------------------------------------------------//
} // end namespace gpuip
