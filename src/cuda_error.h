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

#ifndef GPUIP_CUDA_ERROR_H_
#define GPUIP_CUDA_ERROR_H_
#ifdef _WIN32
#  pragma warning (disable : 4065)
#endif
//----------------------------------------------------------------------------//
#include <sstream>
//----------------------------------------------------------------------------//
namespace gpuip {
//----------------------------------------------------------------------------//
inline bool _cudaErrorGetFunction(CUresult c_err, std::string * err,
                                  const std::string & kernel_name)
{
    if (c_err != CUDA_SUCCESS) {
        (*err) += "Cuda: Error, could not find kernel named ";
        (*err) += kernel_name;
        (*err) += "\n";
        switch(c_err) {
            //TODO: add cases here
            default:
                break;
        }
        return true;
    }
    return false;
}
//----------------------------------------------------------------------------//
inline bool _cudaErrorMalloc(cudaError_t c_err, std::string * err)
{
    if (c_err != cudaSuccess) {
        (*err) += "Cuda: error when allocating buffers\n";
        switch(c_err) {
            //TODO: add cases here
            default:
                break;
        }
        return true;
    }
    return false;
}
//----------------------------------------------------------------------------//
inline bool _cudaErrorFree(cudaError_t c_err, std::string * err)
{
    if (c_err != cudaSuccess) {
        (*err) += "Cuda: error when releasing buffers\n";
        switch(c_err) {
            //TODO: add cases here
            default:
                break;
        }
        return true;
    }
    return false;
}
//----------------------------------------------------------------------------//
inline bool _cudaErrorCopy(cudaError_t c_err, std::string * err,
                           const std::string & buffer, Buffer::CopyOperation op)
{
    if (c_err != cudaSuccess) {
        (*err) += "CUDA: error when copying data ";
        (*err) += op == Buffer::COPY_FROM_GPU ? "FROM" : "TO";
        (*err) += " buffer ";
        (*err) += buffer;
        switch(c_err) {
            case cudaErrorInvalidValue:
                (*err) += ". Invalid value.\n";
                break;
            case cudaErrorInvalidDevicePointer:
                (*err) += ". Invalid device pointer.\n";
                break;
            case cudaErrorInvalidMemcpyDirection:
                (*err) += ". Invalid Memcpy direction.\n";
                break;
            case cudaErrorIllegalAddress:
                (*err) += ". Illegal address.\n";
                break;
                //TODO: add cases here
            default: {
                (*err) += ". Unknown error enum: ";
                std::stringstream ss;
                ss << c_err << std::endl;
                (*err) += ss.str();
            }
        }
        return true;
    }
    return false;
}
//----------------------------------------------------------------------------//
inline bool _cudaErrorUnloadModule(CUresult c_err, std::string * err)
{
    if (c_err != CUDA_SUCCESS) {
        (*err) += "Cuda: Error, could not unload module\n";
        switch(c_err) {
            //TODO: add cases here
            default:
                break;
        }
        return true;
    }
    return false;
}
//----------------------------------------------------------------------------//
inline bool _cudaErrorLoadModule(CUresult c_err, std::string * err)
{
    if (c_err != CUDA_SUCCESS) {
        (*err) += "Cuda: Error, could not load module\n";
        switch(c_err) {
            //TODO: add cases here
            default:
                break;
        }
        return true;
    }
    return false;
}
//----------------------------------------------------------------------------//
inline bool _cudaErrorCheckParamSet(CUresult c_err, std::string * err,
                                    const std::string & kernel_name)
{
    if (c_err != CUDA_SUCCESS) {
        (*err) += "Cuda: Error, could not set arguments for kernel named ";
        (*err) += kernel_name;
        (*err) += "\n";
        switch(c_err) {
            //TODO: add cases here
            default:
                break;
        }
        return true;
    }
    return false;
}
//----------------------------------------------------------------------------//
inline bool _cudaErrorParamSetSize(CUresult c_err, std::string * err,
                                   const std::string & kernel_name)
{
    if (c_err != CUDA_SUCCESS) {
        (*err) += "Cuda: Error, could not set arguments size for kernel named ";
        (*err) += kernel_name;
        (*err) += "\n";
        switch(c_err) {
            //TODO: add cases here
            default:
                break;
        }
        return true;
    }
    return false;
}
//----------------------------------------------------------------------------//
inline bool _cudaErrorLaunchKernel(CUresult c_err, std::string * err,
                                   const std::string & kernel_name)
{
    if (c_err != CUDA_SUCCESS) {
        (*err) += "Cuda: Error, could not launch kernel  ";
        (*err) += kernel_name;
        (*err) += "\n";
        switch(c_err) {
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
