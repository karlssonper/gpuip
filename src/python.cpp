#include "gpuip.h"
#include <boost/python.hpp>
#include <boost/numpy.hpp>
//----------------------------------------------------------------------------//
namespace bp = boost::python;
namespace np = boost::numpy;
//----------------------------------------------------------------------------//
class _KernelWrapper : public gpuip::Kernel
{
  public:
    void AddInBuffer(const gpuip::Buffer & buffer)
    {
        this->inBuffers.push_back(buffer.name);
    }

    void AddOutBuffer(const gpuip::Buffer & buffer)
    {
        this->outBuffers.push_back(buffer.name);
    }

    void AddParamInt(const gpuip::Parameter<int> & param)
    {
        this->paramsInt.push_back(param);
    }

    void AddParamFloat(const gpuip::Parameter<float> & param)
    {
        this->paramsFloat.push_back(param);
    }
};
//----------------------------------------------------------------------------//
class _BaseWrapper
{
  public:
    _BaseWrapper(gpuip::GpuEnvironment env)
            : _base(gpuip::Base::Create(env))
    {
        if (_base.get() ==  NULL) {
            throw std::runtime_error("Could not create gpuip base.");
        }
    }

    boost::shared_ptr<_KernelWrapper> CreateKernel(const std::string name)
    {
        gpuip::Kernel::Ptr ptr = _base->CreateKernel(name);
        // safe since KernelWrapper doesnt hold any extra data
        return boost::static_pointer_cast<_KernelWrapper>(ptr);
    }

    void SetDimensions(unsigned int width, unsigned int height)
    {
        _base->SetDimensions(width,height);
    }
    
    void AddBuffer(const gpuip::Buffer & buffer)
    {
        _base->AddBuffer(buffer);
    }

    std::string InitBuffers()
    {
        std::string err;
        _base->InitBuffers(&err);
        return err;
    }

    std::string Build()
    {
        std::string err;
        _base->Build(&err);
        return err;
    }

    std::string Process()
    {
        std::string err;
        _base->Process(&err);
        return err;
    }

    std::string ReadBuffer(const gpuip::Buffer & buffer,
                           np::ndarray & array)
    {
        std::string err;
        _base->Copy(buffer.name, gpuip::Buffer::READ_DATA,
                    array.get_data(), &err);
        return err;
    }

    std::string WriteBuffer(const gpuip::Buffer & buffer,
                            const np::ndarray & array)
    {
        std::string err;
        _base->Copy(buffer.name, gpuip::Buffer::WRITE_DATA,
                    array.get_data(), &err);
        return err;
    }
  private:
    gpuip::Base::Ptr _base;
};
//----------------------------------------------------------------------------//
BOOST_PYTHON_MODULE(pyGpuip)
{
    np::initialize();

    bp::enum_<gpuip::GpuEnvironment>("Environment")
            .value("OpenCL", gpuip::OpenCL)
            .value("CUDA", gpuip::CUDA)
            .value("GLSL", gpuip::GLSL);

    bp::class_<gpuip::Buffer>("Buffer")
            .def_readwrite("name", &gpuip::Buffer::name)
            .def_readwrite("channels", &gpuip::Buffer::channels)
            .def_readwrite("bpp", &gpuip::Buffer::bpp);

    bp::class_<gpuip::Parameter<int> >("ParamInt")
            .def_readwrite("name", &gpuip::Parameter<int>::name)
            .def_readwrite("value", &gpuip::Parameter<int>::value);

    bp::class_<gpuip::Parameter<float> >("ParamFloat")
            .def_readwrite("name", &gpuip::Parameter<float>::name)
            .def_readwrite("value", &gpuip::Parameter<float>::value);

    bp::class_<_KernelWrapper, boost::shared_ptr<_KernelWrapper> >("Kernel")
            .def_readwrite("name", &_KernelWrapper::name)
            .def_readwrite("code", &_KernelWrapper::code)
            .def("AddInBuffer", &_KernelWrapper::AddInBuffer)
            .def("AddOutBuffer", &_KernelWrapper::AddOutBuffer)
            .def("AddParamInt", &_KernelWrapper::AddParamInt)
            .def("AddParamFloat", &_KernelWrapper::AddParamFloat);
    
    bp::class_<_BaseWrapper, boost::shared_ptr<_BaseWrapper> >
            ("gpuip",
             bp::init<gpuip::GpuEnvironment>())
            .def("SetDimensions", &_BaseWrapper::SetDimensions)
            .def("AddBuffer", &_BaseWrapper::AddBuffer)
            .def("CreateKernel", &_BaseWrapper::CreateKernel)
            .def("InitBuffers", &_BaseWrapper::InitBuffers)
            .def("Build", &_BaseWrapper::Build)
            .def("Process", &_BaseWrapper::Process)
            .def("ReadBuffer", &_BaseWrapper::ReadBuffer)
            .def("WriteBuffer", &_BaseWrapper::WriteBuffer);
}
//----------------------------------------------------------------------------//
