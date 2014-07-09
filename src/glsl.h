#ifndef GPUIP_GLSL_H_
#define GPUIP_GLSL_H_
//----------------------------------------------------------------------------//
#include "gpuip.h"
#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#else
#include <GL/glew.h>
#include <GL/glut.h>
#endif
//----------------------------------------------------------------------------//
namespace gpuip {
//----------------------------------------------------------------------------//
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
    GLuint _vbo;
    std::vector<GLuint> _fbos;
    std::vector<GLuint> _shaders;
    std::map<std::string, GLuint> _textures;

    bool _DrawQuad(const Kernel & kernel,
                   GLuint fbo,
                   GLuint shader,
                   std::string * error);
};
//----------------------------------------------------------------------------//
} // end namespace gpuip
//----------------------------------------------------------------------------//
#endif
