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

#ifndef GPUIP_IMPL_INTERFACE_H_
#define GPUIP_IMPL_INTERFACE_H_
//----------------------------------------------------------------------------//
#include "gpuip.h"
//----------------------------------------------------------------------------//
namespace gpuip {
//----------------------------------------------------------------------------//
class ImplInterface
{
  public:
    virtual ~ImplInterface() {}

    Buffer::Ptr CreateBuffer(const std::string & name,
                             Buffer::Type type,
                             unsigned int channels);

    Kernel::Ptr CreateKernel(const std::string & name);

    void SetDimensions(unsigned int width, unsigned int height);
    
    virtual double Allocate(std::string * error) = 0;

    virtual double Build(std::string * error) = 0;

    virtual double Run(std::string * error) = 0;

    virtual double Copy(Buffer::Ptr buffer,
                        Buffer::CopyOperation operation,
                        void * data,
                        std::string * error) = 0;
    
    virtual std::string BoilerplateCode(Kernel::Ptr kernel) const = 0;

  protected:
    ImplInterface();
    
    std::map<std::string, Buffer::Ptr> _buffers;
    std::vector<Kernel::Ptr> _kernels;

    unsigned int _w; // width
    unsigned int _h; // height
    
    unsigned int _BufferSize(Buffer::Ptr buffer) const;

};
//----------------------------------------------------------------------------//
} // end namespace gpuip
//----------------------------------------------------------------------------//
#endif
