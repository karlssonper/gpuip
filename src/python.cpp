#include "gpuip.h"
#include "io_wrapper.h"
#include <boost/python.hpp>
#include <boost/numpy.hpp>
//----------------------------------------------------------------------------//
namespace bp = boost::python;
namespace np = boost::numpy;
//----------------------------------------------------------------------------//
namespace gpuip {
//----------------------------------------------------------------------------//
namespace python {
//----------------------------------------------------------------------------//
class BufferWrapper
{
  public:
    BufferWrapper(gpuip::Buffer::Ptr b)
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
class KernelWrapper : public gpuip::Kernel
{
  public:
    void SetInBuffer(const std::string & kernelBufferName,
                     boost::shared_ptr<BufferWrapper> buffer)
    {
        for(size_t i = 0; i < this->inBuffers.size(); ++i) {
            if (this->inBuffers[i].name == kernelBufferName) {
                this->inBuffers[i].buffer = buffer->buffer;
                return;
            }
        }
        this->inBuffers.push_back(
            gpuip::Kernel::BufferLink(buffer->buffer, kernelBufferName));
    }

    void SetOutBuffer(const std::string & kernelBufferName,
                      boost::shared_ptr<BufferWrapper> buffer)
    {
        for(size_t i = 0; i < this->outBuffers.size(); ++i) {
            if (this->outBuffers[i].name == kernelBufferName) {
                this->outBuffers[i].buffer = buffer->buffer;
                return;
            }
        }
        this->outBuffers.push_back(
            gpuip::Kernel::BufferLink(buffer->buffer, kernelBufferName));
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
class ImageProcessorWrapper
{
  public:
    ImageProcessorWrapper(gpuip::GpuEnvironment env)
            : _ip(gpuip::ImageProcessor::Create(env))
    {
        if (_ip.get() ==  NULL) {
            throw std::runtime_error("Could not create gpuip imageProcessor.");
        }
    }

    boost::shared_ptr<KernelWrapper> CreateKernel(const std::string & name)
    {
        gpuip::Kernel::Ptr ptr = _ip->CreateKernel(name);
        // safe since KernelWrapper doesnt hold any extra data
        return boost::static_pointer_cast<KernelWrapper>(ptr);
    }

    boost::shared_ptr<BufferWrapper> CreateBuffer(const std::string & name,
                                                   gpuip::Buffer::Type type,
                                                   unsigned int channels)
    {
        gpuip::Buffer::Ptr ptr = _ip->CreateBuffer(name, type, channels);
        // safe since BufferWrapper doesnt hold any extra data
        return boost::shared_ptr<BufferWrapper>(new BufferWrapper(ptr));
    }

    void SetDimensions(unsigned int width, unsigned int height)
    {
        _ip->SetDimensions(width,height);
    }

    unsigned int Width() const
    {
        return _ip->Width();
    }

    unsigned int Height() const
    {
        return _ip->Height();
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

    std::string ReadBufferFromGPU(boost::shared_ptr<BufferWrapper> buffer)
    {
        std::string err;
        _ip->Copy(buffer->buffer, gpuip::Buffer::COPY_FROM_GPU,
                    buffer->data.get_data(), &err);
        return err;
    }

    std::string WriteBufferToGPU(boost::shared_ptr<BufferWrapper> buffer)
    {
        std::string err;
        _ip->Copy(buffer->buffer, gpuip::Buffer::COPY_TO_GPU,
                    buffer->data.get_data(), &err);
        return err;
    }
    
    std::string BoilerplateCode(boost::shared_ptr<KernelWrapper> k) const
    {
        return _ip->BoilerplateCode(k);
    }
  private:
    gpuip::ImageProcessor::Ptr _ip;
};
//----------------------------------------------------------------------------//
} //end namespace python
//----------------------------------------------------------------------------//
} //end namespace gpuip
//----------------------------------------------------------------------------//
BOOST_PYTHON_MODULE(pygpuip)
{
    namespace gp = gpuip::python;

    np::initialize();

    bp::enum_<gpuip::GpuEnvironment>("Environment")
            .value("OpenCL", gpuip::OpenCL)
            .value("CUDA", gpuip::CUDA)
            .value("GLSL", gpuip::GLSL);

    bp::enum_<gpuip::Buffer::Type>("BufferType")
            .value("UNSIGNED_BYTE", gpuip::Buffer::UNSIGNED_BYTE)
            .value("HALF", gpuip::Buffer::HALF)
            .value("FLOAT", gpuip::Buffer::FLOAT);

    bp::class_<gp::BufferWrapper, boost::shared_ptr<gp::BufferWrapper> >
            ("Buffer", bp::no_init)
            .add_property("name", &gp::BufferWrapper::name)
            .add_property("type", &gp::BufferWrapper::type)
            .add_property("channels", &gp::BufferWrapper::channels) 
            .def_readwrite("data", &gp::BufferWrapper::data)
            .def("Read", &gp::BufferWrapper::Read)
            .def("Read", &gp::BufferWrapper::ReadMT)
            .def("Write", &gp::BufferWrapper::Write)
            .def("Write", &gp::BufferWrapper::WriteMT);
    
    bp::class_<gpuip::Parameter<int> >
            ("ParamInt",bp::init<std::string, int>())
            .def_readonly("name", &gpuip::Parameter<int>::name)
            .def_readwrite("value", &gpuip::Parameter<int>::value);

    bp::class_<gpuip::Parameter<float> >
            ("ParamFloat", bp::init<std::string, float>())
            .def_readonly("name", &gpuip::Parameter<float>::name)
            .def_readwrite("value", &gpuip::Parameter<float>::value);

    bp::class_<gp::KernelWrapper, boost::shared_ptr<gp::KernelWrapper> >
            ("Kernel", bp::no_init)
            .def_readonly("name", &gp::KernelWrapper::name)
            .def_readwrite("code", &gp::KernelWrapper::code)
            .def("SetInBuffer", &gp::KernelWrapper::SetInBuffer)
            .def("SetOutBuffer", &gp::KernelWrapper::SetOutBuffer)
            .def("SetParam", &gp::KernelWrapper::SetParamInt)
            .def("SetParam", &gp::KernelWrapper::SetParamFloat);
    
    bp::class_<gp::ImageProcessorWrapper,
            boost::shared_ptr<gp::ImageProcessorWrapper> >
            ("ImageProcessor",
             bp::init<gpuip::GpuEnvironment>())
            .def("SetDimensions", &gp::ImageProcessorWrapper::SetDimensions)
            .add_property("width", &gp::ImageProcessorWrapper::Width)
            .add_property("height", &gp::ImageProcessorWrapper::Height)
            .def("CreateBuffer", &gp::ImageProcessorWrapper::CreateBuffer)
            .def("CreateKernel", &gp::ImageProcessorWrapper::CreateKernel)
            .def("Allocate", &gp::ImageProcessorWrapper::Allocate)
            .def("Build", &gp::ImageProcessorWrapper::Build)
            .def("Run", &gp::ImageProcessorWrapper::Run)
            .def("ReadBufferFromGPU",
                 &gp::ImageProcessorWrapper::ReadBufferFromGPU)
            .def("WriteBufferToGPU",
                 &gp::ImageProcessorWrapper::WriteBufferToGPU)
            .def("BoilerplateCode",
                 &gp::ImageProcessorWrapper::BoilerplateCode);

    bp::def("CanCreateGpuEnvironment",&gpuip::ImageProcessor::CanCreate);
}
//----------------------------------------------------------------------------//
