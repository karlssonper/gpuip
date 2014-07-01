#include "gpuip.h"
#include "base.h"

namespace gpuip {

Base * CreateOpenCL(unsigned int w, unsigned int h);
Base * CreateCUDA(unsigned int w, unsigned int h);
Base * CreateGLSL(unsigned int w, unsigned int h);

Base * Factory::Create(GpuEnvironment env, unsigned int w, unsigned int h)
{
    switch(env) {
        case OpenCL:
            return CreateOpenCL(w,h);
        case CUDA:
            return CreateCUDA(w,h);
        case GLSL:
            return CreateGLSL(w,h);
        default:
            std::cerr << "gpuip error: Could not create env" << std::endl;
            return NULL;
    }
}


} // end namespace gpuip
