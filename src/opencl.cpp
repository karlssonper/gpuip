#include "opencl.h"
#include "opencl_error.h"
#include <ctime>
//----------------------------------------------------------------------------//
namespace gpuip {
//----------------------------------------------------------------------------//
ImageProcessor * CreateOpenCL()
{
    return new OpenCLImpl();
}
//----------------------------------------------------------------------------//
OpenCLImpl::OpenCLImpl()
        : ImageProcessor(OpenCL)
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
    _queue = clCreateCommandQueue(_ctx, _device_id,
                                  CL_QUEUE_PROFILING_ENABLE, NULL);
}
//----------------------------------------------------------------------------//
OpenCLImpl::~OpenCLImpl()
{
    std::string err;
    if (!_ReleaseBuffers(&err)) {
        std::cerr << err << std::endl;
    }

    err.clear();
    if (!_ReleaseKernels(&err)) {
        std::cerr << err << std::endl;
    }
    
    clReleaseCommandQueue(_queue);
    clReleaseContext(_ctx);
}
//----------------------------------------------------------------------------//
double OpenCLImpl::Allocate(std::string * err)
{
    const std::clock_t start = std::clock();

    if(!_ReleaseBuffers(err)) {
        return GPUIP_ERROR;
    }

    cl_int cl_err;
    std::map<std::string,Buffer::Ptr>::const_iterator it;
    for (it = _buffers.begin(); it != _buffers.end(); ++it) {
        _clBuffers[it->second->name] = clCreateBuffer(
            _ctx, CL_MEM_READ_WRITE,
            _BufferSize(it->second), NULL, &cl_err);
        if (_clErrorInitBuffers(cl_err, err)) {
            return GPUIP_ERROR;
        }
    }
    return ( std::clock() - start ) / (long double) CLOCKS_PER_SEC;
}
//----------------------------------------------------------------------------//
double OpenCLImpl::Build(std::string * error)
{
    const std::clock_t start = std::clock();

    if(!_ReleaseKernels(error)) {
        return GPUIP_ERROR;
    }
    
    cl_int cl_err;
    for(size_t i = 0; i < _kernels.size(); ++i) {
        const char * code = _kernels[i]->code.c_str();
        const char * name = _kernels[i]->name.c_str();
        cl_program program = clCreateProgramWithSource(
            _ctx, 1, &code, NULL,  &cl_err);
        if (_clErrorCreateProgram(cl_err, error)) {
            return GPUIP_ERROR;
        }
        
        // Build program
        cl_err = clBuildProgram(program, 1, &_device_id, NULL, NULL, NULL);
        if (_clErrorBuildProgram(cl_err, error, program, _device_id, name)) {
            return GPUIP_ERROR;
        }
    
        // Create kernel from program
        _clKernels.push_back(clCreateKernel(program, name, &cl_err));
        if (_clErrorCreateKernel(cl_err, error)) {
            return GPUIP_ERROR;
        }
    }
    return ( std::clock() - start ) / (long double) CLOCKS_PER_SEC;
}
//----------------------------------------------------------------------------//
double OpenCLImpl::Run(std::string * err)
{
    cl_event event;
    for(size_t i = 0; i < _kernels.size(); ++i) {
        if (!_EnqueueKernel(*_kernels[i].get(), _clKernels[i], event, err)) {
            return GPUIP_ERROR;
        }
    }
    clFinish(_queue);
    clWaitForEvents(1, &event);
    cl_ulong start,end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                            sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                            sizeof(cl_ulong), &end, NULL);
    return (double)(end-start) * 1.0e-6 ;
}
//----------------------------------------------------------------------------//
double OpenCLImpl::Copy(Buffer::Ptr buffer,
                        Buffer::CopyOperation op,
                        void * data,
                        std::string * error)
{
    cl_event event;
    cl_int cl_err = CL_SUCCESS; //set to success to get rid of compiler warnings
    if (op == Buffer::COPY_FROM_GPU) {
        cl_err =  clEnqueueReadBuffer(
            _queue,  _clBuffers[buffer->name],
            CL_TRUE /* function call returns when copy is done */ ,
            0, _BufferSize(buffer), data, 0 , NULL, &event);
    } else if (op == Buffer::COPY_TO_GPU) {
        cl_err =  clEnqueueWriteBuffer(
            _queue,  _clBuffers[buffer->name],
            CL_TRUE /* function call returns when copy is done */ ,
            0, _BufferSize(buffer), data, 0 , NULL, &event);
    }
    if (_clErrorCopy(cl_err, error, buffer->name, op)) {
        return GPUIP_ERROR;
    }
    clWaitForEvents(1, &event);
    cl_ulong start,end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                            sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                            sizeof(cl_ulong), &end, NULL);
    return (double)(end-start) * 1.0e-6 ;
}
//----------------------------------------------------------------------------//
bool OpenCLImpl::_EnqueueKernel(const Kernel & kernel,
                                const cl_kernel & clKernel,
                                cl_event & event,
                                std::string * err)
{
    cl_int cl_err;
    cl_int argc = 0;
    
    // Set kernel arguments in the following order:
    // 1. Input buffers.
    const size_t size = sizeof(cl_mem);
    for(size_t j = 0; j < kernel.inBuffers.size(); ++j) {
        cl_err = clSetKernelArg(clKernel, argc++, size,
                                &_clBuffers[kernel.inBuffers[j].buffer->name]);
    }

    // 2. Output buffers.
    for(size_t j = 0; j < kernel.outBuffers.size(); ++j) {
        cl_err = clSetKernelArg(clKernel, argc++, size,
                                &_clBuffers[kernel.outBuffers[j].buffer->name]);
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
        return GPUIP_ERROR;
    }
    
    const size_t global_work_size[] = { _w, _h };    
    cl_err = clEnqueueNDRangeKernel(_queue, clKernel, 2, NULL,
                                    global_work_size, NULL, 0, NULL, &event);

    if (_clErrorEnqueueKernel(cl_err, err, kernel)) {
        return false;
    }
        
    return true;
}
//----------------------------------------------------------------------------//
inline std::string _GetTypeStr(const Buffer::Ptr & buffer)
{
    std::stringstream type;
    switch(buffer->type) {
        case Buffer::UNSIGNED_BYTE:
            type << "uchar";
            break;
        case Buffer::HALF:
            type << "half";
            break;
        case Buffer::FLOAT:
            type << "float";
            break;
        default:
            type << "float";
    };
    
    // Half vector type is not always supported
    // instead of half4 * data, we have to use half * data
    if (buffer->channels > 1 && buffer->type != Buffer::HALF) {
        type << buffer->channels;
    }
    return type.str();
}
//----------------------------------------------------------------------------//
std::string OpenCLImpl::BoilerplateCode(Kernel::Ptr kernel) const
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
        const std::string & name = kernel->inBuffers[i].buffer->name;
        ss << "__global const " << _GetTypeStr(_buffers.find(name)->second)
           << " * " << kernel->inBuffers[i].name
           << (_buffers.find(name)->second->type == Buffer::HALF ? "_half" : "");
    }
    for(size_t i = 0; i < kernel->outBuffers.size(); ++i) {
        ss << (first ? "" : indent);
        first = false;
        const std::string & name = kernel->outBuffers[i].buffer->name;
        ss << "__global " <<  _GetTypeStr(_buffers.find(name)->second)
           << " * " << kernel->outBuffers[i].name
           << (_buffers.find(name)->second->type == Buffer::HALF ? "_half" : "");
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

    // Do half to float conversions (if needed)
    for(size_t i = 0; i < kernel->inBuffers.size(); ++i) {
        const std::string & bname = kernel->inBuffers[i].buffer->name;
        Buffer::Ptr buf = _buffers.find(bname)->second;
        if (buf->type == Buffer::HALF) {
            if (!i) {
                ss << "    // half to float conversion\n";
            }
            
            if (buf->channels == 1) {
                ss << "    const float " << kernel->inBuffers[i].name
                   << " = vload_half(idx, "
                   << kernel->inBuffers[i].name << "_half);\n";
                continue;
            }
            
            std::stringstream subss;
            subss << "    const float" << buf->channels << " "
                  << kernel->inBuffers[i].name
                  << " = (float" << buf->channels << ")(";
            const std::string preindent = subss.str();
            subss.str("");
            subss << ",\n" << std::string(preindent.size(), ' ');
            const std::string subindent = subss.str();
            ss << preindent;
            bool subfirst = true;
            for (unsigned int j = 0; j < buf->channels; ++j) {
                ss << (subfirst ? "" : subindent);
                subfirst = false;
                ss << "vload_half(" << buf->channels << " * idx + " << j << ", "
                   << kernel->inBuffers[i].name << "_half)";
            }
            ss <<");\n";

            if (i == kernel->inBuffers.size() - 1) {
                ss << "\n";
            }
        }
    }

    ss << "    // kernel code\n";
    for (size_t i = 0; i < kernel->outBuffers.size(); ++i) {
        Buffer::Ptr b = _buffers.find(
            kernel->outBuffers[i].buffer->name)->second;
        ss << "    ";
        if (b->type == Buffer::HALF) {
            ss << "float" << b->channels << " ";
        }
        ss << kernel->outBuffers[i].name;
        if (b->type != Buffer::HALF) {
            ss << "[idx]";
        }
        ss << " = ";
        if (b->channels == 1) {
            ss << "0;\n";
        } else {
            ss << "(";
            if (b->type != Buffer::HALF) {
                ss << _GetTypeStr(b);
            } else {
                ss << "float" << b->channels;
            }
            ss << ")(";
            for (size_t j = 0; j < b->channels; ++j) {
                ss << (j ==0 ? "" : ", ") << "0";
            }
            ss << ");\n";
        }
    }

    // Do half to float conversions (if needed)
    for(size_t i = 0; i < kernel->outBuffers.size(); ++i) {
        const std::string & bname = kernel->outBuffers[i].buffer->name;
        Buffer::Ptr buf = _buffers.find(bname)->second;
        if (buf->type == Buffer::HALF) {
            if (!i) {
                ss << "\n    // float to half conversion\n";
            }

            for (unsigned int j = 0; j < buf->channels; ++j) {
                ss << "    vstore_half(" << kernel->outBuffers[i].name;
                switch(j) {
                    case 0:
                        ss << (buf->channels > 1 ? ".x" : "");
                        break;
                    case 1:
                        ss << ".y";
                        break;
                    case 2:
                        ss << ".z";
                        break;
                    case 3:
                        ss << ".w";
                        break;
                }
                        
                ss << ", ";
                if (buf->channels > 1) {
                    ss << buf->channels << " * ";
                }
                ss << "idx + " << j << ", "
                   << kernel->outBuffers[i].name << "_half);\n";
            }
        }
    }
    
    ss << "}";
    
    return ss.str();
}
//----------------------------------------------------------------------------//
bool OpenCLImpl::_ReleaseBuffers(std::string * err)
{
    std::map<std::string,  cl_mem>::iterator itb;
    for(itb = _clBuffers.begin(); itb != _clBuffers.end(); ++itb) {
        cl_int cl_err = clReleaseMemObject(itb->second);
        if (_clErrorReleaseMemObject(cl_err, err)) {
            return false;
        }
    }
    _clBuffers.clear();
    return true;
}
//----------------------------------------------------------------------------//
bool OpenCLImpl::_ReleaseKernels(std::string * err)
{
    for(size_t i = 0; i < _clKernels.size(); ++i) {
        cl_int cl_err = clReleaseKernel(_clKernels[i]);
        if (_clErrorReleaseKernel(cl_err, err)) {
            return false;
        }
    }
    _clKernels.clear();
    return true;
}
//----------------------------------------------------------------------------//
} // end namespace gpuip
