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

#include "opencl.h"
#include "opencl_error.h"
#include <ctime>
//----------------------------------------------------------------------------//
// Plugin interface
extern "C" GPUIP_DECLSPEC gpuip::ImplInterface * CreateImpl()
{
    return new gpuip::OpenCLImpl();
}
extern "C" GPUIP_DECLSPEC void DeleteImpl(gpuip::ImplInterface * impl)
{
    delete impl;
}
//----------------------------------------------------------------------------//
namespace gpuip {
//----------------------------------------------------------------------------//
bool _SetBufferArgs(const std::vector<Kernel::BufferLink> & bls,
                    const std::map<std::string, cl_mem> & clBuffers,
                    const cl_kernel & clKernel,
                    cl_int & argc,
                    std::string * err);
//----------------------------------------------------------------------------//
template<typename T>
bool _SetParamArgs(const std::vector<Parameter<T> > & params,
                   const cl_kernel & clKernel,
                   cl_int & argc,
                   std::string * err);
//----------------------------------------------------------------------------//
std::string _GetTypeStr(const Buffer::Ptr & buffer);
//----------------------------------------------------------------------------//
void _BoilerplateBufferArgs(
    std::stringstream & ss,
    const std::vector<Kernel::BufferLink> & bufferLinks,
    const std::string & indent,
    int & argcount,
    bool input);
//----------------------------------------------------------------------------//
template<typename T>
void _BoilerplateParamArgs(std::stringstream & ss,
                           const std::vector<Parameter<T > > & params,
                           const char * typenameStr,
                           const std::string & indent,
                           int & argcount);
//----------------------------------------------------------------------------//
void _BoilerplateHalfToFloat(
    std::stringstream & ss,
    const std::vector<Kernel::BufferLink> & bufferLinks);
//----------------------------------------------------------------------------//
void _BoilerplateKernelCode(
    std::stringstream & ss,
    const std::vector<Kernel::BufferLink> & bufferLinks);
//----------------------------------------------------------------------------//
void _BoilerplateFloatToHalf(
    std::stringstream & ss,
    const std::vector<Kernel::BufferLink> & bufferLinks);
//----------------------------------------------------------------------------//
OpenCLImpl::OpenCLImpl()
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
        const Buffer::Ptr & b = it->second;
        _clBuffers[b->Name()] = clCreateBuffer(
            _ctx, CL_MEM_READ_WRITE,  _BufferSize(b), NULL, &cl_err);
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
        const char * code = _kernels[i]->Code().c_str();
        const char * name = _kernels[i]->Name().c_str();
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
            _queue,  _clBuffers[buffer->Name()],
            CL_TRUE /* function call returns when copy is done */ ,
            0, _BufferSize(buffer), data, 0 , NULL, &event);
    } else if (op == Buffer::COPY_TO_GPU) {
        cl_err =  clEnqueueWriteBuffer(
            _queue,  _clBuffers[buffer->Name()],
            CL_TRUE /* function call returns when copy is done */ ,
            0, _BufferSize(buffer), data, 0 , NULL, &event);
    }
    if (_clErrorCopy(cl_err, error, buffer->Name(), op)) {
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
    if (!_SetBufferArgs(kernel.InputBuffers(), _clBuffers,clKernel,argc,err)) {
        return false;
    }
    
     // 2. Output buffers.
    if(!_SetBufferArgs(kernel.OutputBuffers(), _clBuffers,clKernel,argc,err)) {
        return false;
    }
    
    // 3. Int parameters
    if(!_SetParamArgs(kernel.ParamsInt(), clKernel, argc, err)) {
        return false;
    }

    // 4. Float parameters
    if(!_SetParamArgs(kernel.ParamsFloat(), clKernel, argc, err)) {
        return false;
    }

    // Set width and height parameters
    cl_err = clSetKernelArg(clKernel, argc++, sizeof(int),&_w);
    cl_err = clSetKernelArg(clKernel, argc++, sizeof(int),&_h);

    // It should be fine to check once all the arguments have been set
    if (_clErrorSetKernelArg(cl_err, err, kernel.Name())) {
        return false;
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
std::string OpenCLImpl::BoilerplateCode(Kernel::Ptr kernel) const
{
    std::stringstream ss;

    // Indent string (used to indent arguments)
    ss << ",\n" << std::string(kernel->Name().size() + 1, ' ');
    const std::string indent = ss.str();
    ss.str(""); //clears the sstream

    // Header with arguments
    ss << "__kernel void\n" << kernel->Name() << "(";
    int argcount = 0;
    _BoilerplateBufferArgs(ss, kernel->InputBuffers(), indent, argcount, true);
    _BoilerplateBufferArgs(ss, kernel->OutputBuffers(),indent,argcount, false);
    _BoilerplateParamArgs(ss, kernel->ParamsInt(), "int", indent,argcount);
    _BoilerplateParamArgs(ss, kernel->ParamsFloat(), "float", indent,argcount);
    ss << indent << "const int width"
       << indent << "const int height)\n";

    // Code for index and dimension check
    ss << "{\n";
    ss << "    const int x = get_global_id(0);\n";
    ss << "    const int y = get_global_id(1);\n\n";
    ss << "    // array index\n";
    ss << "    const int idx = x + width * y;\n\n";
    ss << "    // inside image bounds check\n";
    ss << "    if (x >= width || y >= height) {\n";
    ss << "        return;\n";
    ss << "    }\n\n";

     // Do half to float conversions  on input buffers (if needed)
    _BoilerplateHalfToFloat(ss, kernel->InputBuffers());

    // Starting kernel code, writing single value or vectors to all zero
    _BoilerplateKernelCode(ss, kernel->OutputBuffers());

    // Do float to half conversions on output buffers (if needed)
    _BoilerplateFloatToHalf(ss, kernel->OutputBuffers());
    
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
bool _SetBufferArgs(const std::vector<Kernel::BufferLink> & bls,
                    const std::map<std::string, cl_mem> & clBuffers,
                    const cl_kernel & clKernel,
                    cl_int & argc,
                    std::string * err)
{
    const size_t size = sizeof(cl_mem);
    for(size_t i = 0; i < bls.size(); ++i) {
        const Buffer::Ptr & targetBuffer = bls[i].TargetBuffer();
        const cl_mem * clBuffer = &clBuffers.find(targetBuffer->Name())->second;
        cl_int cl_err = clSetKernelArg(clKernel, argc++, size, clBuffer);
        if (_clErrorSetKernelArg(cl_err, err, bls[i].Name())) {
            return false;
        }
    }
    return true;
}
//----------------------------------------------------------------------------//
template<typename T>
bool _SetParamArgs(const std::vector<Parameter<T> > & params,
                   const cl_kernel & clKernel,
                   cl_int & argc,
                   std::string * err)
{
    for(size_t i = 0; i < params.size(); ++i) {
        T value = params[i].Value();
        cl_int cl_err = clSetKernelArg(clKernel, argc++, sizeof(T), &value);
        if (_clErrorSetKernelArg(cl_err, err, params[i].Name())) {
            return false;
        }
    }
    return true;
}
//----------------------------------------------------------------------------//
std::string _GetTypeStr(const Buffer::Ptr & buffer)
{
    std::stringstream type;
    switch(buffer->Type()) {
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
    if (buffer->Channels() > 1 && buffer->Type() != Buffer::HALF) {
        type << buffer->Channels();
    }
    return type.str();
}
//----------------------------------------------------------------------------//
void _BoilerplateBufferArgs(
    std::stringstream & ss,
    const std::vector<Kernel::BufferLink> & bufferLinks,
    const std::string & indent,
    int & argcount,
    bool input)
{
    for(size_t i = 0; i < bufferLinks.size(); ++i) {
        //if first argument, don't indent
        if(!argcount++) {
            ss << indent;
        }

        ss << "__global ";
        
        // Only input buffers are declared const
        if(input) {
            ss << "const ";
        }

        const Buffer::Ptr & buffer = bufferLinks[i].TargetBuffer();
        
        // Actual pointer
        ss << _GetTypeStr(buffer)  << " * " << bufferLinks[i].Name();

        // Rename if half
        if(buffer->Type() == Buffer::HALF) {
            ss << "_half";
        }
    }
}
//----------------------------------------------------------------------------//
template<typename T>
void _BoilerplateParamArgs(std::stringstream & ss,
                           const std::vector<Parameter<T > > & params,
                           const char * typenameStr,
                           const std::string & indent,
                           int & argcount)
{
    for(size_t i = 0; i < params.size(); ++i) {
        if (!argcount++) {
            ss << indent;
        }
        ss << "const " << typenameStr << " " << params[i].Name();
    }
}
//----------------------------------------------------------------------------//
void _BoilerplateHalfToFloat(
    std::stringstream & ss,
    const std::vector<Kernel::BufferLink> & bufferLinks)
{
    int count = 0;
    for(size_t i = 0; i < bufferLinks.size(); ++i) {
        const Buffer::Ptr & buf = bufferLinks[i].TargetBuffer();
        const std::string & name = bufferLinks[i].Name();
        // if buffer is of type half, do conversion
        if (buf->Type() == Buffer::HALF) {
            if (!count++) { // only place comment once
                ss << "    // half to float conversion\n";
            }

            // Only 1 channel is a specialcase
            if (buf->Channels() == 1) {
                ss << "    const float " << name << " = "
                   << " = vload_half(idx, " << name << "_half);\n";
                continue;
            }

            // const floatX = make_floatX( 
            std::stringstream subss;
            subss << "    const float" << buf->Channels() << " "
                  << bufferLinks[i].Name()
                  << " = (float" << buf->Channels() << ")(";
            const std::string preindent = subss.str();
            ss << preindent;
            
            // Indent for every row in the make_floatX call
            subss.str("");
            subss << ",\n" << std::string(preindent.size(), ' ');
            const std::string subindent = subss.str();
            
            for (unsigned int j = 0; j < buf->Channels(); ++j) {
                if(!j) { // only subindent after first row
                    ss << subindent;
                }

                // half2float conversion from single value unsigned short
                ss << "vload_half("
                   << buf->Channels() << " * idx + " << j
                   << ", " << name << "_half)";
            }
            ss <<");\n";
        }
    }

    // If any half to float conversions were added, end with a new line
    if (count) {
        ss << "\n";
    }
}
//----------------------------------------------------------------------------//
void _BoilerplateKernelCode(
    std::stringstream & ss,
    const std::vector<Kernel::BufferLink> & bufferLinks)
{
    ss << "    // kernel code\n";
    for (size_t i = 0; i < bufferLinks.size(); ++i) {
        Buffer::Ptr b = bufferLinks[i].TargetBuffer();
        ss << "    ";

        // If half, we need to declare a new variable
        if (b->Type() == Buffer::HALF) {
            ss << "float";

            // If more than 1 channel, then it's a vector
            if(b->Channels() > 1) {
                ss << b->Channels();
            }
            ss << " ";
        }

        // variale name
        ss << bufferLinks[i].Name();

        // if not half, then we write directly to the global memory
        if (b->Type() != Buffer::HALF) {
            ss << "[idx]";
        }
        
        ss << " = ";
        if (b->Channels() == 1) { // single value -> just a zero
            ss << "0;\n";
        } else { // if not single value -> vector
            ss << "make_";
            if (b->Type() != Buffer::HALF) {
                ss << _GetTypeStr(b);
            } else {
                ss << "float" << b->Channels();
            }

            // populate vector with zeros
            ss << "(";
            for (size_t j = 0; j < b->Channels(); ++j) {
                ss << (j ==0 ? "" : ", ") << "0";
            }
            ss << ");\n";
        }
    }
}
//----------------------------------------------------------------------------//
void _BoilerplateFloatToHalf(
    std::stringstream & ss,
    const std::vector<Kernel::BufferLink> & bufferLinks)
{
    // Do half to float conversions (if needed)
    int count = 0;
    for(size_t i = 0; i < bufferLinks.size(); ++i) {
        const Buffer::Ptr & buf = bufferLinks[i].TargetBuffer();
        const std::string name = bufferLinks[i].Name();

        // if buffer pixel type is half, do conversion
        if (buf->Type() == Buffer::HALF) {
            //only write count once
            if (!count++) {
                    ss << "\n    // float to half conversion\n";
            }

            // need to write every channel individually
            for (unsigned int j = 0; j < buf->Channels(); ++j) {
                // variable name (defined in arguments
                ss << "    vstore_half(" << name;


                switch(j) {
                    case 0:
                        ss << (buf->Channels() > 1 ? ".x" : "");
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
                if (buf->Channels() > 1) {
                    ss << buf->Channels() << " * ";
                }
                ss << "idx + " << j << ", "
                   << name << "_half);\n";
            }
        }
    }
}
//----------------------------------------------------------------------------//
} // end namespace gpuip
//----------------------------------------------------------------------------//
