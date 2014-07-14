#ifndef GPUIP_H_
#define GPUIP_H_
//----------------------------------------------------------------------------//
#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <stdexcept>
//----------------------------------------------------------------------------//
#ifdef _GPUIP_PYTHON_BINDINGS
#include <boost/shared_ptr.hpp>
#else
#include <tr1/memory>
#endif
//----------------------------------------------------------------------------//
namespace gpuip {
//----------------------------------------------------------------------------//
enum GpuEnvironment { OpenCL, CUDA, GLSL };
//----------------------------------------------------------------------------//
struct Buffer {
    enum CopyOperation{ READ_DATA, WRITE_DATA };
    std::string name;
    unsigned int channels;
    unsigned int bpp; // bytes per pixel
};
//----------------------------------------------------------------------------//
template<typename T>
struct Parameter
{
    std::string name;
    T value;
};
//----------------------------------------------------------------------------//
struct Kernel {
#ifdef _GPUIP_PYTHON_BINDINGS
    typedef boost::shared_ptr<Kernel> Ptr;
#else // else _GPUIP_PYTHON_BINDINGS
    typedef std::tr1::shared_ptr<Kernel> Ptr;
#endif // end _GPUIP_PYTHON_BINDINGS
        
    std::string name;
    std::string code;
    std::vector<std::string> inBuffers;
    std::vector<std::string> outBuffers;
    std::vector<Parameter<int> > paramsInt;
    std::vector<Parameter<float> > paramsFloat;
};
//----------------------------------------------------------------------------//
class Base
{
  public:
#ifdef _GPUIP_PYTHON_BINDINGS
    typedef boost::shared_ptr<Base> Ptr;
#else
    typedef std::tr1::shared_ptr<Base> Ptr;
#endif
    
    static Ptr Create(GpuEnvironment env);
    
    virtual ~Base() {}
   
    GpuEnvironment GetGpuEnvironment() const
    {
        return _env;
    }

    bool AddBuffer(const Buffer & buffer);

    Kernel::Ptr CreateKernel(const std::string & name);
    
    Kernel::Ptr GetKernel(const std::string & name);

    void SetDimensions(unsigned int width, unsigned int height);
    
    virtual bool InitBuffers(std::string * err) = 0;

    virtual bool Build(std::string * err) = 0;

    virtual bool Process(std::string * err) = 0;

    virtual bool Copy(const std::string & buffer,
                      Buffer::CopyOperation op,
                      void * data,
                      std::string * err) = 0;
               
  protected:
    Base(GpuEnvironment env);

    GpuEnvironment _env;
    unsigned int _w; // width
    unsigned int _h; // height
    std::map<std::string, Buffer> _buffers;
    std::vector<Kernel::Ptr> _kernels;

    unsigned int _GetBufferSize(const Buffer & buffer) const
    {
        return buffer.bpp * _w * _h;
    }
    
  private:
    Base();
    Base(const Base &);
    void operator=(const Base &);
};
//----------------------------------------------------------------------------//
} //end namespace gpuip
//----------------------------------------------------------------------------//
#endif
