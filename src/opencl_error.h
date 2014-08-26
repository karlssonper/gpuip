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

#ifndef GPUIP_OPENCL_ERROR_H_
#define GPUIP_OPENCL_ERROR_H_
#include <sstream>
//----------------------------------------------------------------------------//
namespace gpuip {
//----------------------------------------------------------------------------//
inline bool _clErrorEnqueueKernel(cl_int cl_err, std::string * err,
                                  const gpuip::Kernel & kernel)
{
    if (cl_err != CL_SUCCESS) {
        (*err) += "OpenCL: error when enqueuing kernel ";
        (*err) += kernel.name;
        std::cout << cl_err << std::endl;
        switch(cl_err) {
            case CL_INVALID_KERNEL_ARGS: {
                (*err) += ". Invalid kernel arguments. The gpuip kernel has the"
                        " following data:\n";
                std::stringstream ss;
                ss << "In buffers: ";
                for (size_t i = 0; i < kernel.inBuffers.size(); ++i) {
                    ss << kernel.inBuffers[i].name << "("
                       << kernel.inBuffers[i].buffer->name << "), ";
                }
                ss << "\n";
                
                ss << "Out buffers: ";
                for (size_t i = 0; i < kernel.outBuffers.size(); ++i) {
                    ss << kernel.outBuffers[i].name << "("
                       << kernel.outBuffers[i].buffer->name << "), ";
                }
                ss << "\n";

                ss << "Parameters int: ";
                for (size_t i = 0; i < kernel.paramsInt.size(); ++i) {
                    ss << "(" << kernel.paramsInt[i].name << ","
                       << kernel.paramsInt[i].value << "), ";
                }
                ss << "\n";

                ss << "Parameters float: ";
                for (size_t i = 0; i < kernel.paramsFloat.size(); ++i) {
                    ss << "(" << kernel.paramsFloat[i].name << ","
                       << kernel.paramsFloat[i].value << "), ";
                }
                ss << "\n";
                (*err) += ss.str();
                break;
            }
            default:
                break;
        }
        return true;
    }
    return false;
}
//----------------------------------------------------------------------------//
inline bool _clErrorInitBuffers(cl_int cl_err, std::string * err)
{
    if (cl_err != CL_SUCCESS) {
        (*err) += "OpenCL: error when creating buffers\n";
        switch(cl_err) {
            //TODO: add cases here
            default:
                break;
        }
        return true;
    }
    return false;
}
//----------------------------------------------------------------------------//
inline bool _clErrorReleaseMemObject(cl_int cl_err, std::string * err)
{
    if (cl_err != CL_SUCCESS) {
        (*err) += "OpenCL: error when releasing buffers\n";
        switch(cl_err) {
            //TODO: add cases here
            default:
                break;
        }
        return true;
    }
    return false;
}
//----------------------------------------------------------------------------//
inline bool _clErrorReleaseKernel(cl_int cl_err, std::string * err)
{
    if (cl_err != CL_SUCCESS) {
        (*err) += "OpenCL: error when releasing kernel\n";
        switch(cl_err) {
            //TODO: add cases here
            default:
                break;
        }
        return true;
    }
    return false;
}
//----------------------------------------------------------------------------//
inline bool _clErrorCreateProgram(cl_int cl_err, std::string * err)
{
    if (cl_err != CL_SUCCESS) {
        (*err) += "OpenCL: error when creating program\n";
        switch(cl_err) {
            //TODO: add cases here
            default:
                break;
        }
        return true;
    }
    return false;
}
//----------------------------------------------------------------------------//
inline bool _clErrorBuildProgram(cl_int cl_err, std::string * err,
                                 cl_program p, cl_device_id device_id,
                                 std::string kernel_name)
{
    if (cl_err != CL_SUCCESS) {
        (*err) += "OpenCL: error when building kernel: ";
        (*err) += kernel_name;
        (*err) += "\n\n";
        char buf[0x10000];
        clGetProgramBuildInfo(p, device_id, CL_PROGRAM_BUILD_LOG,
                              0x10000, buf, NULL);
        (*err) += std::string(buf);
        return true;
    }
    return false;
}
//----------------------------------------------------------------------------//
inline bool _clErrorCreateKernel(cl_int cl_err, std::string * err)
{
    if (cl_err != CL_SUCCESS) {
        (*err) += "OpenCL: error when creating kernel\n";
        switch(cl_err) {
            //TODO: add cases here
            default:
                break;
        }
        return true;
    }
    return false;
}
//----------------------------------------------------------------------------//
inline bool _clErrorCopy(cl_int cl_err, std::string * err,
                         const std::string & buffer, Buffer::CopyOperation op)
{
    if (cl_err != CL_SUCCESS) {
        (*err) += "OpenCL: error when copying data ";
        (*err) += op == Buffer::COPY_FROM_GPU ? "FROM" : "TO";
        (*err) += " buffer";
        (*err) += buffer;
        switch(cl_err) {
            case CL_INVALID_MEM_OBJECT:
                (*err) += ". Invalid memory object. Does the buffer exist and "
                        "has it been created? "
                        "(i.e. gpuip::ImageProcessor::CreateBuffer).";
            default:
                break;
        }
        return true;
    }
    return false;
}
//----------------------------------------------------------------------------//
inline bool _clErrorSetKernelArg(cl_int cl_err, std::string * err,
                                 const std::string & kernel_name)
{
    if (cl_err != CL_SUCCESS) {
        (*err) += "OpenCL: error in argument setup in kernel ";
        (*err) += kernel_name;
        switch(cl_err) {
            //TODO: add cases here
            default:
                break;
        }
        return true;
    }
    return false;
}
//----------------------------------------------------------------------------//
} // end namespace gpuip
//----------------------------------------------------------------------------//
#endif
