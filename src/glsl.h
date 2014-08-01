#ifndef GPUIP_GLSL_H_
#define GPUIP_GLSL_H_
//----------------------------------------------------------------------------//
#include "gpuip.h"
#include <GL/glew.h>
//----------------------------------------------------------------------------//
namespace gpuip {
//----------------------------------------------------------------------------//
class GLSLImpl : public Base
{
  public:
    GLSLImpl();

    virtual bool InitBuffers(std::string * err);
      
    virtual bool Build(std::string * err);

    virtual bool Process(std::string * err);
    
    virtual bool Copy(const std::string & buffer,
                      Buffer::CopyOperation op,
                      void * data,
                      std::string * err);

    virtual std::string GetBoilerplateCode(Kernel::Ptr kernel) const;
    
  protected:
    bool _glewInit;
    GLuint _vbo;
    GLuint _rboId;
    GLuint _vertexShaderID;
    std::vector<GLuint> _fbos;
    std::vector<GLuint> _programs;
    std::map<std::string, GLuint> _textures;

    bool _DrawQuad(const Kernel & kernel,
                   GLuint fbo,
                   GLuint program,
                   std::string * error);

    bool _InitGLEW(std::string * err);
};
//----------------------------------------------------------------------------//
} // end namespace gpuip
//----------------------------------------------------------------------------//
#endif
