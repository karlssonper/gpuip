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
    std::map<std::string, CUdeviceptr> _cudaBuffers;
    std::map<std::string, CUtexref> _cudaTextures;
    std::map<std::string, size_t> _cudaPitch;
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
