#include "glsl.h"

namespace gpuip {

Base * CreateGLSL(unsigned int width, unsigned int height)
{
    return new GLSLImpl(width, height);
}

GLSLImpl::GLSLImpl(unsigned int width, unsigned int height)
        : Base(gpuip::GLSL, width, height)
{
}

bool GLSLImpl::InitBuffers(std::string * err)
{
}

bool GLSLImpl::Build(std::string * err)
{
}

bool GLSLImpl::Process(std::string * err)
{
}

bool GLSLImpl::Copy(const std::string & buffer,
                    Buffer::CopyOperation op,
                    void * data,
                    std::string * err)
{
}

} // end namespace gpuip
