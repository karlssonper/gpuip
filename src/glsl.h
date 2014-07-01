#ifndef GPUIP_GLSL_H_
#define GPUIP_GLSL_H_

#include "base.h"

namespace gpuip {

class GLSLImpl : public Base
{
  public:
    GLSLImpl(unsigned int width, unsigned int height);

    virtual bool InitBuffers(std::string * err);
      
    virtual bool Build(std::string * err);

    virtual bool Process(std::string * err);
    
    virtual bool Copy(const std::string & buffer,
                      Buffer::CopyOperation op,
                      void * data,
                      std::string * err);

  protected:
    
};

} // end namespace gpuip

#endif
