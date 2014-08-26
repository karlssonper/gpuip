/*
The MIT License (MIT)

Copyright (c) 2014 Per Karlsson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

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
