#ifndef GPUIP_CUDA_ERROR_H_
#define GPUIP_CUDA_ERROR_H_
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
