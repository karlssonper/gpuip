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
//
#define GPUIP_ERROR -1.0
//----------------------------------------------------------------------------//
enum GpuEnvironment { OpenCL, CUDA, GLSL };
//----------------------------------------------------------------------------//
struct Buffer {
#ifdef _GPUIP_PYTHON_BINDINGS
    typedef boost::shared_ptr<Buffer> Ptr;
#else
    typedef std::tr1::shared_ptr<Buffer> Ptr;
#endif
    enum CopyOperation{ READ_DATA, WRITE_DATA };
    enum Type{ UNSIGNED_BYTE, HALF, FLOAT };
    Buffer(const std::string & name, Type type, unsigned int channels);
    std::string name;
    Type type;
    unsigned int channels;
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
#else
    typedef std::tr1::shared_ptr<Kernel> Ptr;
#endif
    Kernel(const std::string & name);
    std::string name;
    std::string code;
    std::vector<std::pair<Buffer::Ptr,std::string> > inBuffers;
    std::vector<std::pair<Buffer::Ptr,std::string> > outBuffers;
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

    static bool CanCreateGpuEnvironment(GpuEnvironment env);
    
    GpuEnvironment GetGpuEnvironment() const
    {
        return _env;
    }

    Buffer::Ptr CreateBuffer(const std::string & name,
                             Buffer::Type type,
                             unsigned int channels);

    Kernel::Ptr CreateKernel(const std::string & name);
    
    Kernel::Ptr GetKernel(const std::string & name);

    void SetDimensions(unsigned int width, unsigned int height);

    unsigned int GetWidth() const
    {
        return _w;
    }

    unsigned int GetHeight() const
    {
        return _h;
    }

    virtual double Allocate(std::string * err) = 0;

    virtual double Build(std::string * err) = 0;

    virtual double Process(std::string * err) = 0;

    virtual double Copy(const std::string & buffer,
                      Buffer::CopyOperation op,
                      void * data,
                      std::string * err) = 0;

    virtual std::string GetBoilerplateCode(Kernel::Ptr kernel) const = 0;
               
  protected:
    Base(GpuEnvironment env);

    GpuEnvironment _env;
    unsigned int _w; // width
    unsigned int _h; // height
    std::map<std::string, Buffer::Ptr> _buffers;
    std::vector<Kernel::Ptr> _kernels;

    unsigned int _GetBufferSize(Buffer::Ptr buffer) const;
  
  private:
    Base();
    Base(const Base &);
    void operator=(const Base &);
};
//----------------------------------------------------------------------------//
} //end namespace gpuip
//----------------------------------------------------------------------------//
#endif
