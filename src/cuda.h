#ifndef GPUIP_CUDA_H_
#define GPUIP_CUDA_H_
//----------------------------------------------------------------------------//
#include "gpuip.h"
#include <cuda.h>
#include <cuda_runtime.h>
//----------------------------------------------------------------------------//
namespace gpuip {
//----------------------------------------------------------------------------//
class CUDAImpl : public ImageProcessor
{
  public:
    CUDAImpl();

    virtual ~CUDAImpl();
    
    virtual double Allocate(std::string * err);
    
    virtual double Build(std::string * err);

    virtual double Run(std::string * err);
    
    virtual double Copy(Buffer::Ptr buffer,
                        Buffer::CopyOperation op,
                        void * data,
                        std::string * err);

    virtual std::string BoilerplateCode(Kernel::Ptr kernel) const;
    
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

    bool _FreeBuffers(std::string * err);

    bool _UnloadModule(std::string * err);
};
//----------------------------------------------------------------------------//
} // end namespace gpuip

#endif
