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
    std::map<std::string,  cl_mem>::iterator itb;
    for(itb = _clBuffers.begin(); itb != _clBuffers.end(); ++itb) {
        cl_err = clReleaseMemObject(itb->second);
        if (_clErrorReleaseMemObject(cl_err, err)) {
            return false;
        }
    }
    _clBuffers.clear();
    
    std::map<std::string,Buffer>::const_iterator it;
    for (it = _buffers.begin(); it != _buffers.end(); ++it) {
        _clBuffers[it->second.name] = clCreateBuffer(
            _ctx, CL_MEM_READ_WRITE,
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
    // Clear previous kernels if rebuilding
    cl_int cl_err;
    for(size_t i = 0; i < _clKernels.size(); ++i) {
        cl_err = clReleaseKernel(_clKernels[i]);
        if (_clErrorReleaseKernel(cl_err, error)) {
            return false;
        }
    }
    _clKernels.clear();

    for(size_t i = 0; i < _kernels.size(); ++i) {
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
    for(size_t i = 0; i < _kernels.size(); ++i) {
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
    cl_int cl_err = CL_SUCCESS; //set to success to get rid of compiler warnings
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
inline std::string _GetTypeStr(const Buffer & buffer)
{
    std::stringstream type;
    switch(buffer.bpp/buffer.channels) {
        case 1:
            type << "uchar";
            break;
        case 4:
            type << "float";
            break;
        case 8:
            type << "double";
            break;
        default:
            type << "float";
    };
    if (buffer.channels > 1) {
        type << buffer.channels;
    }
    return type.str();
}
std::string OpenCLImpl::GetBoilerplateCode(Kernel::Ptr kernel) const
{
    std::stringstream ss;

    // Indent string
    ss << ",\n" << std::string(kernel->name.size() + 1, ' ');
    const std::string indent = ss.str();
    ss.str(""); //clears the sstream
    
    ss << "__kernel void\n" << kernel->name << "(";

    bool first = true;

    for(size_t i = 0; i < kernel->inBuffers.size(); ++i) {
        ss << (first ? "" : indent);
        first = false;
        const std::string & bname = kernel->inBuffers[i].first.name;
        ss << "__global const " << _GetTypeStr(_buffers.find(bname)->second)
           << " * " << kernel->inBuffers[i].second;
    }
    for(size_t i = 0; i < kernel->outBuffers.size(); ++i) {
        ss << (first ? "" : indent);
        first = false;
        const std::string & bname = kernel->outBuffers[i].first.name;
        ss << "__global " <<  _GetTypeStr(_buffers.find(bname)->second)
           << " * " << kernel->outBuffers[i].second;
    }
    for(size_t i = 0; i < kernel->paramsInt.size(); ++i) {
        ss << (first ? "" : indent);
        first = false;        
        ss << "const int " << kernel->paramsInt[i].name;
    }
    for(size_t i = 0; i < kernel->paramsFloat.size(); ++i) {
        ss << (first ? "" : indent);
        first = false;
        ss << "const float " << kernel->paramsFloat[i].name;
    }
    ss << indent << "const int width" << indent << "const int height)\n";
    
    ss << "{\n";
    ss << "    const int x = get_global_id(0);\n";
    ss << "    const int y = get_global_id(1);\n\n";
    ss << "    // array index\n";
    ss << "    const int idx = x + width * y;\n\n";
    ss << "    // inside image bounds check\n";
    ss << "    if (x >= width || y >= height) {\n";
    ss << "        return;\n";
    ss << "    }\n\n";
    ss << "    // kernel code\n";

    for (size_t i = 0; i < kernel->outBuffers.size(); ++i) {
        const Buffer & b = _buffers.find(
            kernel->outBuffers[i].first.name)->second;
        ss << "    " << kernel->outBuffers[i].second << "[idx] = ";
        if (b.channels == 1) {
            ss << "0;\n";
        } else {
            ss << "make_" << _GetTypeStr(b) << "(";
            for (size_t j = 0; j < b.channels; ++j) {
                ss << (j ==0 ? "" : ", ") << "0";
            }
            ss << ");\n";
        }
    }    
    ss << "}";
    
    return ss.str();
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
    for(size_t j = 0; j < kernel.inBuffers.size(); ++j) {
        cl_err = clSetKernelArg(clKernel, argc++, size,
                                &_clBuffers[kernel.inBuffers[j].first.name]);
    }

    // 2. Output buffers.
    for(size_t j = 0; j < kernel.outBuffers.size(); ++j) {
        cl_err = clSetKernelArg(clKernel, argc++, size,
                                &_clBuffers[kernel.outBuffers[j].first.name]);
    }

    // 3. Int parameters
    for(size_t i = 0; i < kernel.paramsInt.size(); ++i) {
        cl_err = clSetKernelArg(clKernel, argc++, sizeof(int),
                                &kernel.paramsInt[i].value);
    }

    // 4. Float parameters
    for(size_t i = 0; i < kernel.paramsFloat.size(); ++i) {
        cl_err = clSetKernelArg(clKernel, argc++, sizeof(float),
                                &kernel.paramsFloat[i].value);
    }

    // Set width and height parameters
    cl_err = clSetKernelArg(clKernel, argc++, sizeof(int),&_w);
    cl_err = clSetKernelArg(clKernel, argc++, sizeof(int),&_h);

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
