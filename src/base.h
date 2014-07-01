#ifndef GPUIP_BASE_H_
#define GPUIP_BASE_H_

#include "gpuip.h"

namespace gpuip {

class Base
{
  public:
    virtual ~Base() {}
   
    GpuEnvironment GetGpuEnvironment() const
    {
        return _env;
    }

    bool AddBuffer(const Buffer & buffer);

    Kernel * CreateKernel(const std::string & name);
    
    Kernel * GetKernel(const std::string & name);
    
    virtual bool InitBuffers(std::string * err) = 0;

    virtual bool Build(std::string * err) = 0;

    virtual bool Process(std::string * err) = 0;

    virtual bool Copy(const std::string & buffer,
                      Buffer::CopyOperation op,
                      void * data,
                      std::string * err) = 0;
               
  protected:
    Base(GpuEnvironment env, unsigned int width, unsigned int height);

    GpuEnvironment _env;
    
    unsigned int _w; // width
    unsigned int _h; // height

    std::map<std::string, Buffer> _buffers;
    
    std::vector<Kernel> _kernels;

    unsigned int _GetBufferSize(const Buffer & buffer) const
    {
        return buffer.bpp * _w * _h;
    }
    
  private:
    Base();
    Base(const Base &);
    void operator=(const Base &);
};

} // end namespace gpuip

#endif
