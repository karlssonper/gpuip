#ifndef GPUIP_CUDA_H_
#define GPUIP_CUDA_H_
//----------------------------------------------------------------------------//
#include "gpuip.h"
#include <cuda.h>
#include <cuda_runtime.h>
//----------------------------------------------------------------------------//
namespace gpuip {
//----------------------------------------------------------------------------//
class CUDAImpl : public Base
{
  public:
    CUDAImpl();

    virtual double Allocate(std::string * err);
    
    virtual double Build(std::string * err);

    virtual double Process(std::string * err);
    
    virtual double Copy(const std::string & buffer,
                      Buffer::CopyOperation op,
                      void * data,
                      std::string * err);

    virtual std::string GetBoilerplateCode(Kernel::Ptr kernel) const;
    
  protected:
    std::vector<CUfunction> _cudaKernels;
    bool _cudaBuild;
    CUmodule _cudaModule;
    cudaEvent_t _start,_stop;
    std::map<std::string, float*> _cudaBuffers;
    
    bool _LaunchKernel(Kernel & kernel,
                       const CUfunction & cudaKernel,
                       std::string * err);

    void _StartTimer();
    
    double _StopTimer();
};
//----------------------------------------------------------------------------//
} // end namespace gpuip

#endif
