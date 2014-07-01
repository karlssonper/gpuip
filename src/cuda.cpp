#include "cuda.h"

namespace gpuip {

Base * CreateCUDA(unsigned int width, unsigned int height)
{
    return new CUDAImpl(width, height);
}

CUDAImpl::CUDAImpl(unsigned int width, unsigned int height)
        : Base(gpuip::CUDA, width, height)
{
}

bool CUDAImpl::InitBuffers(std::string * err)
{
}

bool CUDAImpl::Build(std::string * err)
{
}

bool CUDAImpl::Process(std::string * err)
{
}

bool CUDAImpl::Copy(const std::string & buffer,
                    Buffer::CopyOperation op,
                    void * data,
                    std::string * err)
{
}

} // end namespace gpuip
