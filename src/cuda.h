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

    virtual bool InitBuffers(std::string * err);
    
    virtual bool Build(std::string * err);

    virtual bool Process(std::string * err);
    
    virtual bool Copy(const std::string & buffer,
                      Buffer::CopyOperation op,
                      void * data,
                      std::string * err);

    virtual std::string GetBoilerplateCode(Kernel::Ptr kernel) const;
    
  protected:
    std::vector<CUfunction> _cudaKernels;
    std::map<std::string, float*> _cudaBuffers;

    void _GetBoilerplateCodeBuffers(
        std::stringstream & ss,
        const std::vector<std::pair<Buffer, std::string> > & buffers,
        const bool inBuffer,
        bool & first,
        const int indent) const;
    
    bool _LaunchKernel(Kernel & kernel,
                       const CUfunction & cudaKernel,
                       std::string * err);
};
//----------------------------------------------------------------------------//
} // end namespace gpuip

#endif
