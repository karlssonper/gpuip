#ifndef GPUIP_CUDA_H_
#define GPUIP_CUDA_H_

#include "base.h"

namespace gpuip {

class CUDAImpl : public Base
{
  public:
    CUDAImpl(unsigned int width, unsigned int height);

    virtual bool InitBuffers(std::string * err);
    
    virtual bool Build(std::string * err);

    virtual bool Process(std::string * err);
    
    virtual bool Copy(const std::string & buffer,
                      Buffer::CopyOperation op,
                      void * data,
                      std::string * err);

  protected:
    
};

} // end namespace gpuip

#endif
