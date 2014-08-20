#ifndef GPUIP_GLSL_H_
#define GPUIP_GLSL_H_
//----------------------------------------------------------------------------//
#include "gpuip.h"
#include <GL/glew.h>
//----------------------------------------------------------------------------//
namespace gpuip {
//----------------------------------------------------------------------------//
class GLSLImpl : public ImageProcessor
{
  public:
    GLSLImpl();

    virtual ~GLSLImpl();

    virtual double Allocate(std::string * err);
      
    virtual double Build(std::string * err);

    virtual double Run(std::string * err);
    
    virtual double Copy(Buffer::Ptr buffer,
                        Buffer::CopyOperation op,
                        void * data,
                        std::string * err);

    virtual std::string BoilerplateCode(Kernel::Ptr kernel) const;
    
  protected:
    bool _glewInit;
    bool _glContextCreated;
    GLint64 _timer;
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

    void _StartTimer();

    double _StopTimer();

    void _DeleteBuffers();
};
//----------------------------------------------------------------------------//
} // end namespace gpuip
//----------------------------------------------------------------------------//
#endif
