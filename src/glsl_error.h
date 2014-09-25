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

#ifndef GPUIP_OPENCL_ERROR_H_
#define GPUIP_OPENCL_ERROR_H_
#ifdef _WIN32
#  pragma warning (disable : 4065)
#endif
//----------------------------------------------------------------------------//
#include <sstream>
//----------------------------------------------------------------------------//
namespace gpuip {
//----------------------------------------------------------------------------//
inline std::string _glErrorToString(GLenum error)
{
    std::stringstream ss;
    //deprecated in mac os
#ifndef __APPLE__
    const GLubyte * errLog = gluErrorString(error);
    ss << errLog;
#endif
    return ss.str();
}
//----------------------------------------------------------------------------//
inline bool _glCheckBuildError(GLuint shader,
                               GLuint vert_shader,
                               GLuint frag_shader,
                               std::string * err)
{
    GLint gl_err;
    glGetProgramiv(shader, GL_LINK_STATUS, &gl_err);
    if (gl_err == 0) {
        std::stringstream ss;
#define ERROR_BUFSIZE 1024
        GLchar errorLog[ERROR_BUFSIZE];
        GLsizei length;

        ss << "GLSL build error.\n";

        glGetShaderInfoLog(vert_shader, ERROR_BUFSIZE, &length, errorLog);
        ss << "Vertex shader errors:\n" << std::string(errorLog, length);
        
        glGetShaderInfoLog(frag_shader, ERROR_BUFSIZE, &length, errorLog);
        ss << "\nFragment shader errors:\n" << std::string(errorLog, length);

        glGetShaderInfoLog(shader, ERROR_BUFSIZE, &length, errorLog);
        ss << "\nLinker errors:\n" << std::string(errorLog, length);

        (*err) += ss.str();
        
        return true;
    }
    return false;
}
//----------------------------------------------------------------------------//
inline bool _glErrorFramebuffer(std::string * err)
{
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if(status != GL_FRAMEBUFFER_COMPLETE) {
        (*err) += "GLSL error: Framebuffer error. ";
        switch(status) {
            case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
                (*err) += "At least one attachment point with a renderbuffer "
                        "or texture attached has its attached object no longer "
                        "in existence or has an attached image with a width or "
                        "height of zero.\n";
                break;
          // This enum doesn't exist in older OpenGL versions.
          //case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS:
          //    (*err) += "Not all attached images have the same width and "
          //          "height.\n";
          //    break;
            case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
                (*err) += "No images are attached to the framebuffer.\n";
            case GL_FRAMEBUFFER_UNSUPPORTED:
                (*err) += "The combination of internal formats of the attached "
                        "images violates an implementation-dependent set of "
                        "restrictions.\n";
                break;
            default:
                break;
        };

        
        return true;
    }
    return false;
}
//----------------------------------------------------------------------------//
inline bool _glErrorCreateTexture(std::string * err)
{
    GLenum gl_err = glGetError();
    if (gl_err != GL_NO_ERROR) {
        (*err) += "GLSL error when creating textures.\n";
        (*err) += _glErrorToString(gl_err);
        return true;
    };
    return false;
}
//----------------------------------------------------------------------------//
inline bool _glErrorDrawSetup(std::string * err,
                              const std::string & kernel_name)
{
    GLenum gl_err = glGetError();
    if (gl_err != GL_NO_ERROR) {
        (*err) += "GLSL error in setup for kernel: ";
        (*err) += kernel_name;
        (*err) += "\n";
        (*err) += _glErrorToString(gl_err);
        return true;
    };
    return false;
}
//----------------------------------------------------------------------------//
inline bool _glErrorDraw(std::string * err, const std::string & kernel_name)
{
    GLenum gl_err = glGetError();
    if (gl_err != GL_NO_ERROR) {
        (*err) += "GLSL error when drawing kernel:";
        (*err) += kernel_name;
        (*err) += "\n";
        (*err) += _glErrorToString(gl_err);
        return true;
    };
    return false;
}
//----------------------------------------------------------------------------//
inline bool _glErrorCopy(std::string * err,
                         const std::string & buffer,
                         Buffer::CopyOperation op)
{
    GLenum gl_err = glGetError();
    if (gl_err != GL_NO_ERROR) {
        (*err) += "GLSL: error when copying data ";
        (*err) += op == Buffer::COPY_FROM_GPU ? "FROM" : "TO";
        (*err) += " buffer ";
        (*err) += buffer;
        (*err) += "\n";
        (*err) += _glErrorToString(gl_err);
        return true;
    }
    return false;
}

} // end namespace gpuip
//----------------------------------------------------------------------------//
#endif
