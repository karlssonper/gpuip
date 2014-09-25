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

#include "glsl.h"
#include "glsl_error.h"
#include "glcontext.h"
#include <string.h>
//----------------------------------------------------------------------------//
// Plugin interface
extern "C" GPUIP_DECLSPEC gpuip::ImplInterface * CreateImpl()
{
    return new gpuip::GLSLImpl();
}
extern "C" GPUIP_DECLSPEC void DeleteImpl(gpuip::ImplInterface * impl)
{
    delete impl;
}
//----------------------------------------------------------------------------//
namespace gpuip {
//----------------------------------------------------------------------------//
inline GLenum _GetType(const Buffer::Ptr & b);
//----------------------------------------------------------------------------//
inline GLenum _GetFormat(const Buffer::Ptr & b);
//----------------------------------------------------------------------------//
inline GLenum _GetInternalFormat(const Buffer::Ptr & b);
//----------------------------------------------------------------------------//
GLSLImpl::GLSLImpl()
        :  _glewInit(false),_glContextCreated(false)
{
}
//----------------------------------------------------------------------------//
GLSLImpl::~GLSLImpl()
{
    _DeleteBuffers();
      
    // Delete shader programs
    for(size_t i = 0; i < _programs.size(); ++i) {
        glDeleteProgram(_programs[i]);
    }
    
    if(_glContextCreated) {
        GLContext::Delete();
    }
}
//----------------------------------------------------------------------------//
bool GLSLImpl::_InitGLEW(std::string * err)
{
    if(!GLContext::Exists()) {
        _glContextCreated = true;
        if(!GLContext::Create(err)) {
            return false;
        }
    }
    
    GLenum result = glewInit();
    if (result != GLEW_OK) {
        std::stringstream ss;
        ss << glewGetErrorString(result) << "\ngpuip could not initiate GLEW\n";
        (*err) += ss.str();
        return GPUIP_ERROR;
    } else {
        _glewInit = true;
    }
    return true;
}
//----------------------------------------------------------------------------//
void GLSLImpl::_StartTimer()
{
    glGetInteger64v(GL_TIMESTAMP, &_timer);
}
//----------------------------------------------------------------------------//
double GLSLImpl::_StopTimer()
{
    const GLint64 timerStart = _timer;
    glFinish();
    glGetInteger64v(GL_TIMESTAMP, &_timer);
    return (_timer - timerStart) / 1000000.0;
}
//----------------------------------------------------------------------------//
void GLSLImpl::_DeleteBuffers()
{
    std::map<std::string, GLuint>::iterator itt;
    if (!_fbos.empty()) {
        glDeleteRenderbuffers(1, &_rboId);
        glDeleteBuffers(_fbos.size(),_fbos.data());
    }
    for(itt = _textures.begin(); itt != _textures.end(); ++itt) {
        glDeleteTextures(1, &itt->second);
    }
    _fbos.clear();
    _textures.clear(); 
}
//----------------------------------------------------------------------------//
double GLSLImpl::Allocate(std::string * err)
{
    if (!_glewInit && !_InitGLEW(err)) {
        return GPUIP_ERROR;
    }

    _StartTimer();

    _DeleteBuffers();
        
    std::map<std::string,Buffer::Ptr>::const_iterator it;
    for(it = _buffers.begin(); it != _buffers.end(); ++it) {
        GLuint texID;
        glGenTextures(1, &texID);
        glBindTexture(GL_TEXTURE_2D, texID);

        // No linear interpolation
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        // Out of bounds -> Value at the border
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        // Mipmaps not needed.
        glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_FALSE); 

        // Allocate memory on gpu (needed to bind framebuffer)
        Buffer::Ptr b = it->second;
        glTexImage2D(GL_TEXTURE_2D, 0, _GetInternalFormat(b),
                     _w, _h, 0, _GetFormat(b), _GetType(b), 0);
        
        _textures[it->second->name] = texID;

        if (_glErrorCreateTexture(err)) {
            return GPUIP_ERROR;
        }
    }
            
    // Create FBOs
    _fbos.reserve(_kernels.size());
    glGenFramebuffers(_kernels.size(), _fbos.data());

    // Create a renderbuffer object to store depth info
    GLuint _rboId;
    glGenRenderbuffers(1, &_rboId);
    glBindRenderbuffer(GL_RENDERBUFFER, _rboId);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, _w, _h);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    // Attach the textures to FBOs color attachment points
    for(size_t i = 0; i < _kernels.size(); ++i) {
        glBindFramebuffer(GL_FRAMEBUFFER, _fbos[i]);
        for(size_t j = 0; j < _kernels[i]->outBuffers.size(); ++j) {
            GLuint texID = _textures[_kernels[i]->outBuffers[j].buffer->name];
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + j, 
                                   GL_TEXTURE_2D, texID, 0 /*mipmap level*/);
        }
       
        // attach the renderbuffer to depth attachment point
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                  GL_RENDERBUFFER, _rboId);
        if (_glErrorFramebuffer(err)) {
            return GPUIP_ERROR;
        }
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    // Create and build quad vbo
    glGenBuffers(1, &_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    float vertices[] = {0,0,1,0,1,1,0,1};
    glBufferData(GL_ARRAY_BUFFER, 32, vertices, GL_STATIC_DRAW);
    
    return _StopTimer();
}
//----------------------------------------------------------------------------//
double GLSLImpl::Build(std::string * err)
{
    if (!_glewInit) {
        _InitGLEW(err);
    }

    _StartTimer();
    
    for(size_t i = 0; i < _programs.size(); ++i) {
        glDeleteProgram(_programs[i]);
    }

    // Simple vert shader code for a quad with texture coordinates
    static const char * vert_shader_code =  
            "#version 120\n"
            "attribute vec2 positionIn;\n"
            "varying vec2 x;\n"
            "void main()\n"
            "{\n"
            "      gl_Position = vec4(vec2(-1) + 2*positionIn,0, 1);\n"
            "      x = positionIn;\n"
            "}";

    _vertexShaderID = glCreateShader(GL_VERTEX_SHADER);
    int length = strlen(vert_shader_code);
    glShaderSource(_vertexShaderID, 1, &vert_shader_code, &length);
    glCompileShader(_vertexShaderID);
    
    _programs.resize(_kernels.size());
    for(size_t i = 0; i < _kernels.size(); ++i) {
        const char * code = _kernels[i]->code.c_str();
        const GLuint fragShaderID = glCreateShader(GL_FRAGMENT_SHADER);
        const int length = strlen(code);
        glShaderSource(fragShaderID, 1, &code, &length);
        glCompileShader(fragShaderID);
  
        _programs[i] = glCreateProgram();
        glAttachShader(_programs[i], fragShaderID);
        glAttachShader(_programs[i], _vertexShaderID);
        glLinkProgram(_programs[i]);
        glDeleteShader(_vertexShaderID);
        glDeleteShader(fragShaderID);
        if(_glCheckBuildError(_programs[i],_vertexShaderID,fragShaderID, err)) {
            return GPUIP_ERROR;
        }
    }

    return _StopTimer();
}
//----------------------------------------------------------------------------//
double GLSLImpl::Run(std::string * err)
{
    _StartTimer();
    
    glPushAttrib( GL_VIEWPORT_BIT );
    
    // Set the viewport to match the width and height
    glViewport(0, 0, _w, _h);
    
    for(size_t i = 0; i < _kernels.size(); ++i) {
        if (!_DrawQuad(*_kernels[i].get(), _fbos[i], _programs[i], err)) {
            return GPUIP_ERROR;
        }
    }

    // Reset back to the previous viewport
    glPopAttrib();
    return _StopTimer();
}
//----------------------------------------------------------------------------//
double GLSLImpl::Copy(Buffer::Ptr b,
                      Buffer::CopyOperation op,
                      void * data,
                      std::string * err)
{
    _StartTimer();
    if (op == Buffer::COPY_FROM_GPU) {
        glBindTexture(GL_TEXTURE_2D, _textures[b->name]);
        glGetTexImage(GL_TEXTURE_2D, 0, _GetFormat(b), _GetType(b), data);
    } else if (op == Buffer::COPY_TO_GPU) {
        glBindTexture(GL_TEXTURE_2D, _textures[b->name]);
        glTexImage2D(GL_TEXTURE_2D, 0, _GetInternalFormat(b),
                     _w, _h, 0, _GetFormat(b), _GetType(b), data);
    }
    if (_glErrorCopy(err, b->name, op)) {
        return GPUIP_ERROR;
    }
    return _StopTimer();
}
//----------------------------------------------------------------------------//
std::string GLSLImpl::BoilerplateCode(Kernel::Ptr kernel) const
{
    std::stringstream ss;
    ss << "#version 120\n";
    for(size_t i = 0; i < kernel->inBuffers.size(); ++i) {
        ss << "uniform sampler2D " << kernel->inBuffers[i].name << ";\n";
    }
    for(size_t i = 0; i < kernel->paramsInt.size(); ++i) {
        ss << "uniform int " << kernel->paramsInt[i].name <<";\n";
    }
    for(size_t i = 0; i < kernel->paramsFloat.size(); ++i) {
        ss << "uniform float " << kernel->paramsFloat[i].name <<";\n";
    }
    ss << "varying vec2 x; // texture coordinates\n"
       << "uniform float dx; // delta\n\n"
       << "void main()\n"
       << "{\n";
    for(size_t i = 0; i < kernel->outBuffers.size(); ++i) {
        if (i) {
            ss << "\n";
        }
        ss << "    // gl_FragData[" << i << "] is buffer "
           << kernel->outBuffers[i].name << "\n"
           << "    gl_FragData[" << i<<"] = vec4(0,0,0,1);\n";  
    }    
    ss << "}";
    return ss.str();
}
//----------------------------------------------------------------------------//
bool GLSLImpl::_DrawQuad(const Kernel & kernel,
                         GLuint fbo,
                         GLuint program,
                         std::string * error)
{   
    // Bind framebuffer and clear previous content
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Tell OpenGL how many buffers to draw
    std::vector<GLenum> enums;
    for(size_t i = 0; i < kernel.outBuffers.size(); ++i) {
        enums.push_back(GL_COLOR_ATTACHMENT0 + i);
    }
    glDrawBuffers(enums.size(), &enums[0]);

    // Load program (kernel)
    glUseProgram(program);

    // Bind vbo
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    
    // Coordinates for drawing the quad
    GLint loc = glGetAttribLocation(program, "positionIn");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 2, GL_FLOAT, 0, 8, 0);
       
    // Uniform data setup
    for(size_t i = 0; i < kernel.paramsInt.size(); ++i) {
        loc = glGetUniformLocation(program, kernel.paramsInt[i].name.c_str());
        glUniform1i(loc, kernel.paramsInt[i].value);
    }
    for(size_t i = 0; i < kernel.paramsFloat.size(); ++i) {
        loc = glGetUniformLocation(program, kernel.paramsFloat[i].name.c_str());
        glUniform1f(loc, kernel.paramsFloat[i].value);
    }
    glUniform1f(glGetUniformLocation(program, "dx"), 1.0f/_w);

    // Save current active texture 
    GLint activeTexture;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &activeTexture);

    // Texture setup
    for(size_t i = 0; i < kernel.inBuffers.size(); ++i) {
        loc = glGetUniformLocation(program, kernel.inBuffers[i].name.c_str());
        glUniform1i(loc, i);
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, _textures[kernel.inBuffers[i].buffer->name]);
    }

    if (_glErrorDrawSetup(error, kernel.name)) {
        return false;
    }

    // Draw quad
    glDrawArrays(GL_QUADS, 0, 4);
    
    if (_glErrorDraw(error, kernel.name)) {
        return false;
    }
    
    // Unload framebuffer, vbo and program
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glUseProgram(0); 
    
    // Back to old active texture
    glActiveTexture(activeTexture);

    return true;
}
//----------------------------------------------------------------------------//
GLenum _GetType(const Buffer::Ptr & b)
{
    switch(b->type) {
        case Buffer::UNSIGNED_BYTE:
            return GL_UNSIGNED_BYTE;
        case Buffer::HALF:
            return GL_HALF_FLOAT;
        case Buffer::FLOAT:
            return GL_FLOAT;
        default:
            std::cerr << "gpuip: Unknown OpenGL data type.. " << std::endl;
            return GL_FLOAT;
    }
}
//----------------------------------------------------------------------------//
GLenum _GetFormat(const Buffer::Ptr & b)
{
    switch(b->channels) {
        case 1:
            return GL_RED;
        case 2:
            return GL_RG;
        case 3:
            return GL_RGB;
        case 4:
            return GL_RGBA;
        default:
            std::cerr << "gouip: Unknown OpenGL buffer channels " << b->channels
                      << std::endl;
            return GL_RGB;
    }
}
//----------------------------------------------------------------------------//
GLenum _GetInternalFormat(const Buffer::Ptr & b)
{
    const GLenum type = _GetType(b);
    const GLenum format = _GetFormat(b);
    if (type == GL_HALF_FLOAT) {
        switch(format){
            case GL_RED:
                return GL_R16F;
            case GL_RG:
                return GL_RG16F;
            case GL_RGB:
                return GL_RGB16F;
            case GL_RGBA:
                return GL_RGBA16F;
            default:
                break;
        }
    } else if (type == GL_FLOAT) {
        switch(format){
            case GL_RED:
                return GL_R32F;
            case GL_RG:
                return GL_RG32F;
            case GL_RGB:
                return GL_RGB32F;
            case GL_RGBA:
                return GL_RGBA32F;
            default:
                break;
        }
    }
    return format;
}
//----------------------------------------------------------------------------//
} // end namespace gpuip
