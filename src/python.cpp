#include "gpuip.h"
#include "io_wrapper.h"
#include <boost/python.hpp>
#include <boost/numpy.hpp>
//----------------------------------------------------------------------------//
namespace bp = boost::python;
namespace np = boost::numpy;
//----------------------------------------------------------------------------//
class _BufferWrapper
{
  public:
    _BufferWrapper(gpuip::Buffer::Ptr b)
            : buffer(b),
              data(np::empty(
                  bp::make_tuple(0,0),
                  np::dtype::get_builtin<unsigned char>())) {}
              
    std::string name() const
    {
        return buffer->name;
    }

    gpuip::Buffer::Type type() const
    {
        return buffer->type;
    }

    unsigned int channels() const
    {
        return buffer->channels;
    }
    
    std::string Read(const std::string & filename)
    {
        return ReadMT(filename, 0);
    }
    
    std::string ReadMT(const std::string & filename, int numThreads)
    {
        std::string err;
        gpuip::io::ReadFromFile(&data, *buffer.get(), filename, numThreads);
        return err;
    }
    
    std::string Write(const std::string & filename)
    {
        return WriteMT(filename, 0);
    }

    std::string WriteMT(const std::string & filename, int numThreads)
    {
        std::string err;
        gpuip::io::WriteToFile(&data, *buffer.get(), filename, numThreads);
        return err;
    }
    
    gpuip::Buffer::Ptr buffer;
    np::ndarray data;
};
//----------------------------------------------------------------------------//
class _KernelWrapper : public gpuip::Kernel
{
  public:
    void SetInBuffer(const std::string & kernelBufferName,
                     boost::shared_ptr<_BufferWrapper> buffer)
    {
        for(size_t i = 0; i < this->inBuffers.size(); ++i) {
            if (this->inBuffers[i].second == kernelBufferName) {
                this->inBuffers[i].first = buffer->buffer;
                return;
            }
        }
        this->inBuffers.push_back(make_pair(buffer->buffer, kernelBufferName));
    }

    void SetOutBuffer(const std::string & kernelBufferName,
                      boost::shared_ptr<_BufferWrapper> buffer)
    {
        for(size_t i = 0; i < this->outBuffers.size(); ++i) {
            if (this->outBuffers[i].second == kernelBufferName) {
                this->outBuffers[i].first = buffer->buffer;
                return;
            }
        }
        this->outBuffers.push_back(make_pair(buffer->buffer, kernelBufferName));
    }

    void SetParamInt(const gpuip::Parameter<int> & param)
    {
        for(size_t i = 0 ; i < this->paramsInt.size(); ++i) {
            if (this->paramsInt[i].name == param.name) {
                this->paramsInt[i].value = param.value;
                return;
            }
        }
        this->paramsInt.push_back(param);
    }

    void SetParamFloat(const gpuip::Parameter<float> & param)
    {
        for(size_t i = 0 ; i < this->paramsFloat.size(); ++i) {
            if (this->paramsFloat[i].name == param.name) {
                this->paramsFloat[i].value = param.value;
                return;
            }
        }
        this->paramsFloat.push_back(param);
    }
};
//----------------------------------------------------------------------------//
class _ImageProcessorWrapper
{
  public:
    _ImageProcessorWrapper(gpuip::GpuEnvironment env)
            : _ip(gpuip::ImageProcessor::Create(env))
    {
        if (_ip.get() ==  NULL) {
            throw std::runtime_error("Could not create gpuip imageProcessor.");
        }
    }

    boost::shared_ptr<_KernelWrapper> CreateKernel(const std::string & name)
    {
        gpuip::Kernel::Ptr ptr = _ip->CreateKernel(name);
        // safe since KernelWrapper doesnt hold any extra data
        return boost::static_pointer_cast<_KernelWrapper>(ptr);
    }

    boost::shared_ptr<_BufferWrapper> CreateBuffer(const std::string & name,
                                                   gpuip::Buffer::Type type,
                                                   unsigned int channels)
    {
        gpuip::Buffer::Ptr ptr = _ip->CreateBuffer(name, type, channels);
        // safe since BufferWrapper doesnt hold any extra data
        return boost::shared_ptr<_BufferWrapper>(new _BufferWrapper(ptr));
    }

    void SetDimensions(unsigned int width, unsigned int height)
    {
        _ip->SetDimensions(width,height);
    }

    std::string Allocate()
    {
        std::string err;
        _ip->Allocate(&err);
        return err;
    }

    std::string Build()
    {
        std::string err;
        _ip->Build(&err);
        return err;
    }

    std::string Run()
    {
        std::string err;
        _ip->Run(&err);
        return err;
    }

    std::string ReadBufferFromGPU(boost::shared_ptr<_BufferWrapper> buffer)
    {
        std::string err;
        _ip->Copy(buffer->name(), gpuip::Buffer::READ_DATA,
                    buffer->data.get_data(), &err);
        return err;
    }

    std::string WriteBufferToGPU(boost::shared_ptr<_BufferWrapper> buffer)
    {
        std::string err;
        _ip->Copy(buffer->name(), gpuip::Buffer::WRITE_DATA,
                    buffer->data.get_data(), &err);
        return err;
    }
    
    std::string GetBoilerplateCode(boost::shared_ptr<_KernelWrapper> k) const
    {
        return _ip->GetBoilerplateCode(k);
    }
  private:
    gpuip::ImageProcessor::Ptr _ip;
};
//----------------------------------------------------------------------------//
BOOST_PYTHON_MODULE(pygpuip)
{
    np::initialize();

    bp::enum_<gpuip::GpuEnvironment>("Environment")
            .value("OpenCL", gpuip::OpenCL)
            .value("CUDA", gpuip::CUDA)
            .value("GLSL", gpuip::GLSL);

    bp::enum_<gpuip::Buffer::Type>("BufferType")
            .value("UNSIGNED_BYTE", gpuip::Buffer::UNSIGNED_BYTE)
            .value("HALF", gpuip::Buffer::HALF)
            .value("FLOAT", gpuip::Buffer::FLOAT);

    bp::class_<_BufferWrapper, boost::shared_ptr<_BufferWrapper> >
            ("Buffer", bp::no_init)
            .add_property("name", &_BufferWrapper::name)
            .add_property("type", &_BufferWrapper::type)
            .add_property("channels", &_BufferWrapper::channels) 
            .def_readwrite("data", &_BufferWrapper::data)
            .def("Read", &_BufferWrapper::Read)
            .def("Read", &_BufferWrapper::ReadMT)
            .def("Write", &_BufferWrapper::Write)
            .def("Write", &_BufferWrapper::WriteMT);
    
    bp::class_<gpuip::Parameter<int> >("ParamInt")
            .def_readwrite("name", &gpuip::Parameter<int>::name)
            .def_readwrite("value", &gpuip::Parameter<int>::value);

    bp::class_<gpuip::Parameter<float> >("ParamFloat")
            .def_readwrite("name", &gpuip::Parameter<float>::name)
            .def_readwrite("value", &gpuip::Parameter<float>::value);

    bp::class_<_KernelWrapper, boost::shared_ptr<_KernelWrapper> >
            ("Kernel", bp::no_init)
            .def_readwrite("name", &_KernelWrapper::name)
            .def_readwrite("code", &_KernelWrapper::code)
            .def("SetInBuffer", &_KernelWrapper::SetInBuffer)
            .def("SetOutBuffer", &_KernelWrapper::SetOutBuffer)
            .def("SetParam", &_KernelWrapper::SetParamInt)
            .def("SetParam", &_KernelWrapper::SetParamFloat);
    
    bp::class_<_ImageProcessorWrapper,
            boost::shared_ptr<_ImageProcessorWrapper> >
            ("ImageProcessor",
             bp::init<gpuip::GpuEnvironment>())
            .def("SetDimensions", &_ImageProcessorWrapper::SetDimensions)
            .def("CreateBuffer", &_ImageProcessorWrapper::CreateBuffer)
            .def("CreateKernel", &_ImageProcessorWrapper::CreateKernel)
            .def("Allocate", &_ImageProcessorWrapper::Allocate)
            .def("Build", &_ImageProcessorWrapper::Build)
            .def("Run", &_ImageProcessorWrapper::Run)
            .def("ReadBufferFromGPU",
                 &_ImageProcessorWrapper::ReadBufferFromGPU)
            .def("WriteBufferToGPU",
                 &_ImageProcessorWrapper::WriteBufferToGPU)
            .def("GetBoilerplateCode",
                 &_ImageProcessorWrapper::GetBoilerplateCode);

    bp::def("CanCreateGpuEnvironment",
            &gpuip::ImageProcessor::CanCreateGpuEnvironment);
}
//----------------------------------------------------------------------------//
